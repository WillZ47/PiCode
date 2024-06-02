from ultralytics import YOLO
import cv2 
import pandas as pd 
import torch
import time

timeNow=time.time()

# Load yolov8 model
model = YOLO('yolov8n.pt')

# Load video
cap = cv2.VideoCapture(0)

detection_list=[]

# Read frames
while True:
    ret, frame = cap.read()

    if ret:
        # Detect objects 
        
        ''' 
        results = model.track(frame, persist=True)
        '''
        results = model(frame)

        # Get detected objects 
        
        #from the results object --> get the boxes attribute 
        #from the boxes attribute--> get the cls attribute (returns a tensor of dtype float32) 
        #turn that object into a list of floats and then map that to int lists sort them, and you get your dataframe

        '''
        detections=pd.Series(list(map(int, results[0].boxes.cls.tolist()))).value_counts()
        detections.index  = detections.index.map(results[0].names)
        ''' 

        detections = pd.Series(list(map(int, results[0].boxes.cls.tolist()))).value_counts().rename(index=results[0].names)


        # Update the index of the series with the new index
   
        print(detections)

        
        detection_list.append(detections)

      

        # Initialize counter for "person" labels


        # Visualize the frame with bounding boxes
        frame_ = results[0].plot()
        cv2.imshow('frame', frame_)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break 

        if time.time()-timeNow>25:
            break

# Release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows() 

print(len(detection_list))