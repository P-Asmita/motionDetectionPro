import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8n-seg.pt') # initializes the YOLOv8 model with the weights provided in the 'yolov8n-seg.pt' file.

#video_path="children.mp4"
capture= cv2.VideoCapture(0) #opens the default camera for video capture.
while capture.isOpened():
    success,frame=capture.read() # captures a frame from the video.

    if success:
        start=time.perf_counter()  #The script calculates the time taken to process the frame using the time.perf_counter() function.
        results=model(frame)

        end=time.perf_counter()
        total_time=end-start
        fps= 1/total_time
        annotated_frame=results[0].plot()

        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("YOLOv8 Inference",annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): #The loop continues until the user presses the 'q' key, at which point the script releases the resources and closes the windows.
            break
    else:
        break
capture.release() #releases the video capture resources.
cv2.destroyAllWindows()
