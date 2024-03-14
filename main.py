import cv2 
import imutils
import os
import face_recognition
import csv
from yoloDet import YoloTRT
from datetime import datetime

KNOWN_FACES_DIR = './known_faces'
TOLERANCE = 0.6
MODEL = 'cnn'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value range is 97 to 122, subtract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

print('Loading known faces...')
known_faces = []
known_names = []

# We organize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):
    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]
        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)

# Initialize YOLO model
model = YoloTRT(library="./models/libmyplugins.so", engine="./models/id-lanyard_detection.engine", conf=0.5, yolo_ver="v5")

# Open CSV file for writing
with open('records.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Name', 'ID', 'Lanyard'])

    # Load the video
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame)
        
        # Face detection and recognition
        locations = face_recognition.face_locations(frame, model=MODEL)
        encodings = face_recognition.face_encodings(frame, locations)
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                
                # Object detection to check if wearing an ID and a Lanyard
                wearing_id = False
                wearing_lanyard = False
                detections, _ = model.Inference(frame)
                for obj in detections:
                    if obj['class'] == 'Card':
                        # Assuming 'Card' represents the ID card class detected by the model
                        wearing_id = True
                    if obj['class'] == 'Lanyard':
                        # Assuming 'Lanyard' represents the lanyard class detected by the model
                        wearing_lanyard = True
                
                # Write to CSV with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, match, wearing_id, wearing_lanyard])
        
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
