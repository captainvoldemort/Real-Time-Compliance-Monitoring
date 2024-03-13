import cv2
import os
import numpy as np

# Create a face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to train the face recognition model
def train_model():
    # Read the label mapping from the file
    label_mapping = {}
    with open('label_mapping.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            user_name = parts[0]
            label = int(parts[1])
            label_mapping[user_name] = label

    # Create lists to store the face samples and their corresponding labels
    faces = []
    labels = []

    # Load the images from the 'Faces' folder
    for file_name in os.listdir('Faces'):
        if file_name.endswith('.jpg'):
            # Extract the user name and PRN from the file name
            parts = file_name.split('_')
            user_name = parts[0]
            prn = parts[1].split('.')[0]

            # Read the image and convert it to grayscale
            image = cv2.imread(os.path.join('Faces', file_name))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Check if a face is detected
            if len(detected_faces) > 0:
                # Crop the detected face region
                face_crop = gray[detected_faces[0][1]:detected_faces[0][1] + detected_faces[0][3],
                                 detected_faces[0][0]:detected_faces[0][0] + detected_faces[0][2]]

                # Append the face sample and label to the lists
                faces.append(face_crop)
                labels.append(label_mapping[user_name])

    # Train the face recognition model using the faces and labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # Save the trained model to a file
    recognizer.save('trained_model.xml')
    return recognizer

# Train the model
Recognizer = train_model()
Recognizer
