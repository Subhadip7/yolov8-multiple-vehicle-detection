import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from tracker import Tracker

# ------------------------------
# Configuration
# ------------------------------
CAPACITY_TONS = 20.0
VEHICLE_WEIGHTS_TONS = {
    'car': 1.5,
    'bus': 12.0,
    'truck': 15.0,
}

# Entry and exit lines (Y coordinates)
ENTRY_LINE_Y = 184
EXIT_LINE_Y = 209
LINE_TOLERANCE = 8
STALE_FRAMES_TO_EVICT = 120  # frames after last seen to evict lost IDs

# ------------------------------
# Optional: Mouse position debug
# ------------------------------
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        # print(point)  # Uncomment for debugging


# ------------------------------
# Setup
# ------------------------------
model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture('tf.mp4')

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Use separate trackers per class for stable IDs
car_tracker = Tracker()
bus_tracker = Tracker()
truck_tracker = Tracker()

# Runtime state
on_bridge_ids = set()  # set of tuples like ('car', id)
id_to_last_cy = {}     # map of ('car', id) -> last center y
current_load_tons = 0.0

# Track last seen frame for each key to evict stale IDs
id_to_last_seen_frame = {}

# Diagnostics (optional)
entries = {'car': 0, 'bus': 0, 'truck': 0}
exits = {'car': 0, 'bus': 0, 'truck': 0}

# Process frames
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    # Process every 3rd frame to reduce load
    if frame_idx % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Run detection
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    # Collect detections by class
    cars, buses, trucks = [], [], []
    for _, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        cls_idx = int(row[5])
        cls_name = class_list[cls_idx] if 0 <= cls_idx < len(class_list) else ''

        if 'car' in cls_name:
            cars.append([x1, y1, x2, y2])
        elif 'bus' in cls_name:
            buses.append([x1, y1, x2, y2])
        elif 'truck' in cls_name:
            trucks.append([x1, y1, x2, y2])

    # Update trackers
    car_boxes = car_tracker.update(cars)
    bus_boxes = bus_tracker.update(buses)
    truck_boxes = truck_tracker.update(trucks)

    # Determine barrier state
    barrier_closed = current_load_tons >= CAPACITY_TONS

    # Draw entry/exit lines (color reflects barrier state)
    entry_color = (0, 0, 255) if barrier_closed else (0, 255, 0)
    exit_color = (0, 0, 255)
    cv2.line(frame, (1, ENTRY_LINE_Y), (1018, ENTRY_LINE_Y), entry_color, 2)
    cv2.line(frame, (3, EXIT_LINE_Y), (1016, EXIT_LINE_Y), exit_color, 2)

    # Helper to process a single detection list
    def process_boxes(boxes, cls_name):
        global current_load_tons
        for bbox in boxes:
            x1, y1, x2, y2, obj_id = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            key = (cls_name, obj_id)
            prev_cy = id_to_last_cy.get(key)

            # Detect crossings using previous and current center Y positions
            if prev_cy is not None:
                crossed_entry = (prev_cy < ENTRY_LINE_Y - LINE_TOLERANCE) and (cy >= ENTRY_LINE_Y + LINE_TOLERANCE)
                crossed_exit = (prev_cy < EXIT_LINE_Y - LINE_TOLERANCE) and (cy >= EXIT_LINE_Y + LINE_TOLERANCE)

                # Handle entry
                if crossed_entry and key not in on_bridge_ids:
                    weight = VEHICLE_WEIGHTS_TONS.get(cls_name, 0.0)
                    # Only allow entry if this vehicle would NOT exceed capacity
                    if current_load_tons + weight <= CAPACITY_TONS:
                        on_bridge_ids.add(key)
                        current_load_tons += weight
                        entries[cls_name] += 1
                    # else: barrier is effectively closing; do not admit

                # Handle exit
                if crossed_exit and key in on_bridge_ids:
                    weight = VEHICLE_WEIGHTS_TONS.get(cls_name, 0.0)
                    if weight > 0:
                        current_load_tons = max(0.0, current_load_tons - weight)
                    on_bridge_ids.discard(key)
                    exits[cls_name] += 1

            # Update last seen Y and last seen frame
            id_to_last_cy[key] = cy
            id_to_last_seen_frame[key] = frame_idx

            # Draw bbox and ID label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{cls_name} #{obj_id}', (x1, max(0, y1 - 10)), 1, 1)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

    # Process detections by class
    process_boxes(car_boxes, 'car')
    process_boxes(bus_boxes, 'bus')
    process_boxes(truck_boxes, 'truck')

    # Evict stale IDs that have not been seen for a while
    to_evict = []
    for key in list(on_bridge_ids):
        last_seen = id_to_last_seen_frame.get(key, 0)
        if frame_idx - last_seen > STALE_FRAMES_TO_EVICT:
            to_evict.append(key)
    for key in to_evict:
        cls_name, _ = key
        weight = VEHICLE_WEIGHTS_TONS.get(cls_name, 0.0)
        if weight > 0:
            current_load_tons = max(0.0, current_load_tons - weight)
        on_bridge_ids.discard(key)

    # Recompute barrier state after any changes
    barrier_closed = current_load_tons >= CAPACITY_TONS

    # Overlay current load and barrier status
    status_text = 'CLOSED' if barrier_closed else 'OPEN'
    status_color = (0, 0, 255) if barrier_closed else (0, 200, 0)
    cvzone.putTextRect(
        frame,
        f'Bridge: {status_text} | Load: {current_load_tons:.1f}/{CAPACITY_TONS:.1f} tons',
        (10, 30),
        1,
        2,
        colorR=status_color,
        colorT=(255, 255, 255)
    )

    # Visual barrier at the entry line when closed
    if barrier_closed:
        cv2.rectangle(frame, (0, max(0, ENTRY_LINE_Y - 6)), (frame.shape[1], ENTRY_LINE_Y + 6), (0, 0, 255), -1)
        cvzone.putTextRect(frame, 'NO ENTRY - OVER CAPACITY', (10, ENTRY_LINE_Y - 35), 1, 1, colorR=(0, 0, 255))

    # Show frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Final stats
print(f"Entries: cars={entries['car']}, buses={entries['bus']}, trucks={entries['truck']}")
print(f"Exits:   cars={exits['car']}, buses={exits['bus']}, trucks={exits['truck']}")
print(f'Final on-bridge count: {len(on_bridge_ids)} | Load: {current_load_tons:.1f} tons')

cap.release()
cv2.destroyAllWindows()