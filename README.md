# **Real-Time Compliance Monitoring**

This repository contains code for a real-time compliance monitoring system designed to mark attendance and check if individuals are wearing ID cards with proper lanyards. The system utilizes face recognition and object detection techniques to achieve its objectives.

## **Project Specifications**

### **Features**

- **Real-Time Monitoring**: The system provides real-time monitoring of compliance, allowing organizations to ensure adherence to safety protocols and regulations.
- **Attendance Marking**: The system automatically marks attendance based on face recognition, eliminating the need for manual attendance tracking.
- **ID Card and Lanyard Detection**: Utilizes object detection to check if individuals are wearing ID cards with proper lanyards, enhancing safety and security measures.
- **User Registration**: Provides a convenient way to register new users into the system by capturing their facial images using the camera.
- **High FPS Inference**: Utilizes optimized YOLOv5 object detection model for high FPS (Frames Per Second) inference, ensuring efficient real-time processing.
- **Scalability**: The system is scalable and can be deployed in various environments such as offices, schools, manufacturing facilities, etc.
- **Easy Integration**: Easily integrates with existing attendance systems or can be used as a standalone solution for compliance monitoring.

### **Files Included**

- **main.py**: This is the main script responsible for running the real-time compliance monitoring system. It utilizes the trained YOLOv5 object detection model to detect objects like ID cards and lanyards and uses face recognition to identify individuals.
- **records.csv**: This CSV file serves as a record of attendance and compliance. It stores information such as the timestamp, individual's name, and whether they were wearing an ID card with a proper lanyard.
- **register_user.py**: This script is used to register new users into the system. It captures 10 images of the user's face using the camera and saves them into a directory within the **`known_faces`** folder.
- **Update register_user.py**: This is an updated version of the **`register_user.py`** script, incorporating improvements or bug fixes.
- **train_yolov5_object_detection.ipynb**: This Jupyter Notebook contains code for training the YOLOv5 object detection model. After training, it generates a **`best.pt`** file, which is used for inference.
- **yoloDet.py**: This Python script contains classes and functions related to the YOLOv5 object detection model, including inference.
- **known_faces/**: Directory containing images of registered users for face recognition. Each user's images are stored in a separate folder named as `PRN_Name-Surname`.

### **Project Workflow**

1. **Training the Object Detection Model**: The **`train_yolov5_object_detection.ipynb`** notebook is used to train the YOLOv5 object detection model. After training, the **`best.pt`** file is generated.
2. **Model Conversion**: The **`best.pt`** file is then copied to a Jetson Nano 4GB Dev Kit. Following the instructions provided in the repository [here](https://github.com/yashreadytobox/JetsonYolov5), the model is converted into a **`.wts`** file and subsequently into an engine file to run inference at high FPS.
3. **Running the Real-Time Monitoring System**: On the Jetson Nano, the **`main.py`** script is executed to run the real-time compliance monitoring system. This script utilizes the trained YOLOv5 model for object detection and face recognition techniques to identify individuals and check for compliance.
4. **Registering New Users**: New users can be registered into the system using the **`register_user.py`** script. This script captures 10 images of the user's face using the camera and saves them into a directory within the **`known_faces`** folder.

## **Usage**

1. **Training the Model**: Run the **`train_yolov5_object_detection.ipynb`** notebook to train the YOLOv5 object detection model.
2. **Model Conversion**: Follow the instructions in the repository [here](https://github.com/yashreadytobox/JetsonYolov5) to convert the trained model into an engine file for inference on the Jetson Nano.
3. **Running the Real-Time Monitoring System**: Execute the **`main.py`** script on the Jetson Nano to run the real-time compliance monitoring system.
4. **Registering New Users**: Use the **`register_user.py`** script to register new users into the system.
