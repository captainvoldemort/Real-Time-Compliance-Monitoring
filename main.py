import cv2
import numpy as np
import pandas as pd
import torch
from deepface import DeepFace
from deepface.basemodels import VGGFace
from yoloDet import YoloTRT

#Face Recognition model
face_model = VGGFace.loadModel()

# Load YOLOv5 model for ID card and lanyard segmentation
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

# Initialize human body and face detection cascades
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=['Person', 'Face Detected', 'Wearing ID Card', 'Wearing Lanyard'])

# Function to detect upper body in a frame and return cropped region
def detect_upper_body(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(bodies) > 0:
        x, y, w, h = bodies[0]  # Consider only the first detected body
        return frame[y:y+h, x:x+w], (x, y, w, h)
    else:
        return None, None

# Function to detect faces in a frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to recognize face using face recognition algorithm
def recognize_face(face_roi, db_path=''):
    # Use DeepFace to find matches in the database
    dfs = DeepFace.find(img=face_roi, db_path=db_path, model_name = 'VGG-Face', model = face_model, distance_metric = 'cosine')
    
    # Extract the name of the most likely match from the first dataframe
    if not dfs:
        return "Unknown"
    else:
        most_likely_match = dfs[0].iloc[0]
        recognized_name = most_likely_match["identity"]
        return recognized_name

# Function to process each frame from video feed
def process_frame(frame):
    # Detect upper body in frame
    upper_body, body_coords = detect_upper_body(frame)
    
    if upper_body is not None:
        # Detect faces in upper body region
        faces = detect_faces(upper_body)
        
        if len(faces) > 0:
            # Get the first detected face
            x, y, w, h = faces[0]
            face_roi = upper_body[y:y+h, x:x+w]
            
            # Recognize face
            person_name = recognize_face(face_roi)
            
            # Perform ID card and lanyard segmentation using YOLOv5
            detections, t = model.Inference(frame)
            # Parse results to check if person is wearing ID card/lanyard
            wearing_id = True  # Placeholder for ID card detection result
            wearing_lanyard = True  # Placeholder for lanyard detection result
            
            # Update results DataFrame
            results_df.loc[len(results_df)] = [person_name, True, wearing_id, wearing_lanyard]
            
            # Save results to CSV file
            results_df.to_csv('results.csv', index=False)
    
    return frame

# Function to process video feed
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame)
        
        cv2.imshow('Processed Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function to start processing
def main():
    # Provide path to video feed
    video_path = 'video_feed.mp4'
    process_video(video_path)

if __name__ == "__main__":
    main()
