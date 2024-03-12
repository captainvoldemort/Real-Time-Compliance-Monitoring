import os
import cv2

def capture_images():
    # Input the name of the person
    person_name = input("Enter the name of the person: ")
    
    # Input the PRN (Personnel Registration Number)
    prn = input("Enter the PRN (Personnel Registration Number): ")
    
    # Create a folder with the person's name and PRN if it doesn't exist
    folder_name = f"{person_name}_{prn}"
    folder_path = os.path.join("dataset", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Capture 5-10 frames containing face images
    num_images = 0
    while num_images < 10:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing image from the camera.")
            break
        
        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Save each detected face as an image in the folder
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            image_name = f"{folder_name}_{num_images}.jpg"
            image_path = os.path.join(folder_path, image_name)
            cv2.imwrite(image_path, face_roi)
            num_images += 1
        
        # Display the frame with face detections
        cv2.imshow("Capture Faces", frame)
        
        # Wait for a key press to capture the next frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to capture images and create the dataset
capture_images()
