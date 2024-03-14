import cv2
import os

# Function to capture images
def capture_images(folder_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Capture 10 images
    print("Capturing images. Please look at the camera and keep still...")
    for i in range(1, 11):
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        
        # Save the image
        image_path = os.path.join(folder_path, f"image_{i}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image {i} captured.")
        
        # Display the image for a brief moment
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(1000)  # Display each image for 1 second
    
    print("Image capture completed.")
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Input PRN, Name, and Surname
    prn = input("Enter PRN: ")
    name = input("Enter Name: ")
    surname = input("Enter Surname: ")
    
    # Create the folder path
    folder_name = f"{prn}_{name}-{surname}"
    folder_path = os.path.join("./known_faces", folder_name)
    
    # Capture images and save to folder
    capture_images(folder_path)

if __name__ == "__main__":
    main()
