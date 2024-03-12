import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='/path/to/your/weights.pt')

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize the image to a square of 640x640 pixels
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
])

# Function to perform inference on a single image
def infer_single_image(image_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    img = transform(img).unsqueeze(0).to(device)  # Add a batch dimension
    
    # Perform inference
    results = model(img)
    
    # Process results
    results.print()  # Print results to console
    results.show()   # Show annotated image
    
    # Return detected objects
    return results.xyxy[0].cpu().numpy()

# Main function
def main():
    image_path = '/path/to/your/image.jpg'  # Replace with your image path
    detections = infer_single_image(image_path)
    print("Detected objects:", detections)

if __name__ == "__main__":
    main()
