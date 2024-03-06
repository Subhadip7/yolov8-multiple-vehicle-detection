# yolov8-multiple-vehicle-detection
This model is very useful to detecting cars, buses, and trucks in a video.

This code snippet is written in Python and uses several libraries (`cv2`, `pandas`, `ultralytics`, `cvzone`) to perform object detection and tracking on a video file. The object detection is carried out using a pre-trained YOLO (You Only Look Once) model, which is a popular method for real-time object detection. Here's a breakdown of the code:

1. Import Libraries:
   - `cv2`: OpenCV library for computer vision tasks.
   - `pandas`: Data manipulation library, here used to handle data frames.
   - `ultralytics`: A company's package that includes the YOLO model.
   - `cvzone`: Computer vision library for easy OpenCV functions.
   - `tracker`: A module that presumably contains a custom tracking class for tracking objects over frames.

2. Load YOLO Model:
   - The YOLO model is loaded with the weights file `'yolov8s.pt'`.

3. Mouse Position Function:
   - A function `RGB` is defined to print the mouse position whenever the mouse moves within a window named 'RGB'.

4. Setup OpenCV Window and Mouse Callback:
   - A named window 'RGB' is created, and the `RGB` function is set to be called whenever a mouse event occurs in this window.

5. Video Capture:
   - The video file `'tf.mp4'` is opened for processing.

6. Read Class Names:
   - Class names for detected objects are read from `'coco.txt'` and stored in a list called `class_list`.

7. Initialize Variables:
   - Counters for frames and different types of vehicles are initialized.
   - A `Tracker` object is created for tracking the vehicles.
   - Two lines (`cy1`, `cy2`) with an `offset` are defined, which the vehicles will cross.

8. Process Video Frames:
   - The video is processed frame by frame in a loop.
   - Every third frame is processed to reduce computational load.
   - Frames are resized for consistency.
   - The YOLO model predicts objects in the frame.
   - The detections are converted into a pandas DataFrame for easier manipulation.

9. Categorize Detected Objects:
   - Detected objects are categorized as cars, buses, or trucks based on the class names and their bounding boxes are stored.

10. Tracking:
    - The `Tracker` object updates the tracked bounding boxes for cars, buses, and trucks.

11. Draw Lines:
    - Two crossing lines are drawn on the frame to count the vehicles when they cross these lines.

12. Count Vehicles:
    - For each type of vehicle (car, bus, truck), if the center of the bounding box crosses the defined line within the specified offset, the respective counter is incremented.

13. Annotate Frames:
    - Bounding boxes and labels for each vehicle are drawn on the frames.
    - The `cvzone.putTextRect` function is used to put text on the frame.

14. Display Frame:
    - The annotated frame is displayed in the 'RGB' window.

15. Exit Condition:
    - If the 'Esc' key is pressed, the loop is broken and the program proceeds to termination.

16. Print Vehicle Counts:
    - After the video processing is complete, the total counts for each type of vehicle are printed.

17. Cleanup:
    - The video capture is released and all OpenCV windows are destroyed.

This script is essentially for a traffic monitoring application, where it counts the number of cars, buses, and trucks passing a certain line in the video. It demonstrates the use of computer vision and machine learning for real-world applications such as traffic analysis and vehicle tracking.







<img width="960" alt="image" src="https://github.com/Subhadip7/yolov8-multiple-vehicle-detection/assets/95004440/6e767950-5c7b-4118-9831-f673db91ad7c">
