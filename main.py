import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
from sklearn.cluster import KMeans


## ========================================================================================
# 1. Output Saver
class VideoOut:
    def __init__(self, output_path, fps=10, width=640, height=340, save_flag=True):
        self.save_flag = save_flag
        if self.save_flag:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            self.width = width
            self.height = height

    def save_frame(self, frame):
        if self.save_flag:
            frame = cv2.resize(frame, (self.width, self.height))
            self.out.write(frame)

    def release(self):
        if self.save_flag:
            self.out.release()


class SaveImg:
    def __init__(self, folder_out="./out_img", save_flag=True):
        os.makedirs(folder_out, exist_ok=True)
        self.i = 0
        self.folder_out = folder_out
        self.width, self.height = 640, 480
        self.save_flag = save_flag

    def save_img(self, frame):
        if self.save_flag:
            frame = cv2.resize(frame, (self.width, self.height))
            path_out = f"{self.folder_out}/img_{self.i}.jpg"
            cv2.imwrite(path_out, frame)
            print(path_out)
            self.i += 1


## ========================================================================================
## 2. Color Detection Class

class ColorDetector:
    def __init__(self):
        # Define color ranges in HSV
        self.color_dict = {
            'black': ([0, 0, 0], [180, 50, 50]),  # Lower max V to keep it dark
            'white': ([0, 0, 220], [180, 40, 255]),  # White should have very low saturation
            'silver': ([0, 0, 180], [180, 30, 240]),  # Distinct from pure white, slightly darker
            'gray': ([0, 0, 50], [180, 20, 180]),  # Separate from black & silver

            'red': ([0, 70, 50], [10, 255, 255]),
            'red2': ([160, 70, 50], [180, 255, 255]),

            'orange': ([10, 100, 50], [24, 255, 255]),
            'yellow': ([15, 100, 120], [40, 255, 255]),  # Ensure it doesn't get mistaken for black

            'green': ([36, 50, 50], [85, 255, 255]),
            'blue': ([90, 70, 70], [130, 255, 255]),  # Lower min S & V so dark blues don't get ignored
            'purple': ([125, 50, 50], [155, 255, 255]),
            'brown': ([10, 100, 20], [30, 255, 150])
        }

    def detect_color_kmeans(self, img, k=3):
        # Convert image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        kernel = np.ones((3, 3), np.uint8)
        hsv = cv2.dilate(hsv, kernel, iterations=4)

        # Convert back to RGB for KMeans
        img_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Reshape for KMeans clustering
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)

        # Apply KMeans to find dominant colors
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get dominant colors
        colors = kmeans.cluster_centers_.astype(np.uint8)

        # Determine most dominant color
        dominant_color = colors[np.argmax(np.bincount(kmeans.labels_))]

        # Convert to HSV for color matching
        hsv_color = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0]

        # Match color
        color_name = self.match_color_hsv(hsv_color)

        return color_name, tuple(map(int, dominant_color[::-1]))  # Return BGR

    def match_color_hsv(self, hsv_color):
        h, s, v = hsv_color

        # Ensure Black Isn't Misclassified
        if v <= 50 and s <= 50:
            return "black"

        # Separate White, Silver, and Grey
        if s <= 50:  # Increased threshold to allow more variations
            if v >= 220:
                return "white"
            elif v >= 170:
                return "silver"  # Lowered threshold so it's detected more
            elif v >= 40:
                return "gray"  # Lowered threshold to capture darker greys

        # Prevent Dark Greys from Becoming Blue
        if 90 <= h <= 130 and s >= 70 and v >= 70:
            return "blue"

        # Match Other Colors
        for color_name, (lower, upper) in self.color_dict.items():
            lower = np.array(lower)
            upper = np.array(upper)

            if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                return color_name

        return "unknown"  # If nothing matches, return "unknown"


## ========================================================================================
## 3. Your Analytics

class YourAnalytics:
    def __init__(self):
        self.class_names = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            8: 'boat',
        }

        # Different colors for different vehicle types
        self.colors = {
            2: (0, 255, 0),  # car: green
            3: (255, 0, 255),  # motorcycle: magenta
            5: (0, 165, 255),  # bus: orange
            7: (0, 0, 255),  # truck: red
            8: (255, 255, 0)  # boat: cyan
        }

        # Initialize the color detector
        self.color_detector = ColorDetector()

    def run(self, frame, yolo_results):
        # Parse detection
        try:
            bbox_results = yolo_results[0].boxes.cpu().numpy()
            arr_box, arr_cls, arr_conf = bbox_results.xyxy.astype(
                int).tolist(), bbox_results.cls.tolist(), bbox_results.conf.tolist()
            yolo_results[0].boxes.id.int().cpu().tolist()
        except:
            return frame

        # Visualization
        for (box, class_id, box_score) in zip(arr_box, arr_cls, arr_conf):
            # Convert class_id to int to use as dict key
            class_id = int(class_id)
            x1, y1, x2, y2 = box

            # Choose color based on class
            if class_id in self.colors:
                color = self.colors[class_id]
            else:
                color = (255, 255, 255)  # white for other objects

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Handle vehicle classes (car, motorcycle, bus, truck, boat)
            if class_id in [2, 3, 5, 7, 8]:
                # Crop the vehicle region (with a small margin to avoid edge effects)
                margin_x = int((x2 - x1) * 0.1)
                margin_y = int((y2 - y1) * 0.1)

                # Ensure margins don't go outside the image
                x1_crop = max(0, x1 + margin_x)
                y1_crop = max(0, y1 + margin_y)
                x2_crop = min(frame.shape[1], x2 - margin_x)
                y2_crop = min(frame.shape[0], y2 - margin_y)

                # Skip if crop dimensions are too small
                vehicle_color_name = "N/A"
                vehicle_color_bgr = None

                if x2_crop - x1_crop > 10 and y2_crop - y1_crop > 10:
                    vehicle_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                    # Detect color
                    vehicle_color_name, vehicle_color_bgr = self.color_detector.detect_color_kmeans(vehicle_crop)

                # Prepare label with class name, confidence, and color
                class_name = self.class_names[class_id]
                label = f"{class_name}: {box_score:.2f}"
                if vehicle_color_name != "N/A":
                    label += f" ({vehicle_color_name})"

                # Draw label background
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Draw a small color sample if we detected a color
                if vehicle_color_bgr is not None:
                    color_sample_size = 15
                    cv2.rectangle(frame, (x1 + label_size[0] + 5, y1 - color_sample_size - 5),
                                  (x1 + label_size[0] + 5 + color_sample_size, y1 - 5),
                                  vehicle_color_bgr, -1)
            else:
                # For non-vehicle classes, simple labeling
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                label = f"{class_name}: {box_score:.2f}"

                # Draw label
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        return frame


## ========================================================================================
## 3. Main Function
def process_video(input_path, output_path, i_video):
    ## 0. Cap
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

        ## 1. Video Out
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_out = VideoOut(output_path, fps=int(fps), width=width, height=height, save_flag=True)
    img_saver = SaveImg(folder_out=f"./out_img/{i_video}/", save_flag=True)

    ## 2. Model
    yolo = YOLO("yolov8x.pt")
    i_frame = 0

    ## 3. Analytics
    analytics = YourAnalytics()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i_frame % 10 != 0:
            i_frame += 1
            continue
        # ----------------------------
        ## Main Infer
        # 1) Detection
        # yolo_results = yolo(frame)
        yolo_results = yolo.track(frame, conf=0.5, classes=[2, 3, 5, 7, 8])

        # 2) Analytics
        frame_out = frame.copy()
        frame_out = analytics.run(frame_out, yolo_results)

        # ----------------------------
        video_out.save_frame(frame_out)
        img_saver.save_img(frame_out)
        i_frame += 1

    cap.release()
    video_out.release()
    if video_out.save_flag:
        print(f"Processed video saved at {output_path}")
    elif not video_out.save_flag:
        print("Set save_flag to True to save your video")

if __name__ == "__main__":
    input_folder = './videos'
    output_folder = './out'

    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    for i_video, video_file in enumerate(video_files):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"processed_{video_file}")
        process_video(input_path, output_path, i_video)