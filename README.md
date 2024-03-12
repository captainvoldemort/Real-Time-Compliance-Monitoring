# **ID CARD VERIFICATION**

## **Overview**

This project aims to develop a real-time human identification and ID card detection system using computer vision techniques. The system is designed to detect human bodies, recognize faces, and identify whether a person is wearing an ID card or lanyard. The project utilizes YOLOv5 for ID card and lanyard segmentation, OpenCV for human body and face detection, and a custom face recognition algorithm.

## **Features**

- Detects human bodies in real-time video feed.
- Recognizes faces within the detected human bodies.
- Utilizes YOLOv5 for ID card and lanyard segmentation.
- Determines whether a person is wearing an ID card or lanyard.
- Provides real-time feedback on the detected individuals.
- Saves results in a CSV file for analysis and logging.

## **Requirements**

- Python 3.x
- OpenCV
- PyTorch
- YOLOv5
- deepFace
- Pandas

## **Installation**

1. Clone the repository to your local machine:
    
    ```bash
    git clone https://github.com/yashreadytobox/ID-CARD-VERIFICATION.git
    ```
    
2. Install the required dependencies: Follow the steps given in the github page below to carry out installations on Jetson Nano 4GB Dev Kit.
https://github.com/yashreadytobox/JetsonYolov5 
3. Download the pre-trained YOLOv5 model and place it in the project directory.
4. Replace the face recognition placeholder with your own face recognition model or algorithm.

## **Usage**
Before running first create a database by registering users through `register_user` script. 

(Make sure that all the paths provided are accurate.)

1. Navigate to the project directory:
    
    ```bash
    cd ID-CARD-VERIFICATION
    ```
    
2. Run the main script to start the human identification system:
    
    ```bash
    python main.py
    ```
    
3. The system will process the video feed from the default camera
4. Press 'q' to exit the processed video feed window.

## **Deployment on Jetson Nano using TensorRT**

1. Convert the trained models to TensorRT format using the NVIDIA TensorRT toolkit.
2. Deploy the converted models on the Jetson Nano device.
3. Integrate the deployment setup with the existing project codebase.

**FOLLOW STEPS GIVEN [HERE](https://github.com/yashreadytobox/JetsonYolov5) TO INSTALL DEPENDANCIES FOR JETSON NANO 4GB DEV KIT**
