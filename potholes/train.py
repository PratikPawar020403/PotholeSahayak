import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def setup_training_data():
    """Setup and verify training data structure"""
    # Define paths (use relative paths)
    data_dir = "data/pothole_dataset/Pothole_Segmentation_YOLOv8"
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    
    # Verify directories exist
    required_dirs = [
        os.path.join(train_dir, "images"),
        os.path.join(train_dir, "labels"),
        os.path.join(valid_dir, "images"),
        os.path.join(valid_dir, "labels")
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Required directory not found: {directory}")
    
    return data_dir

def analyze_dataset(data_dir):
    """Analyze the dataset and print statistics"""
    train_images = os.listdir(os.path.join(data_dir, "train", "images"))
    valid_images = os.listdir(os.path.join(data_dir, "valid", "images"))
    
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of validation images: {len(valid_images)}")
    
    # Analyze image sizes
    train_image_sizes = set()
    valid_image_sizes = set()
    
    for img_name in train_images:
        img_path = os.path.join(data_dir, "train", "images", img_name)
        img = cv2.imread(img_path)
        train_image_sizes.add(img.shape[:2])
    
    for img_name in valid_images:
        img_path = os.path.join(data_dir, "valid", "images", img_name)
        img = cv2.imread(img_path)
        valid_image_sizes.add(img.shape[:2])
    
    print("\nImage size analysis:")
    if len(train_image_sizes) == 1:
        print(f"All training images have the same size: {train_image_sizes.pop()}")
    else:
        print("Training images have varying sizes.")
    
    if len(valid_image_sizes) == 1:
        print(f"All validation images have the same size: {valid_image_sizes.pop()}")
    else:
        print("Validation images have varying sizes.")

def train_model(data_dir):
    """Train the YOLOv8 model"""
    # Load a pretrained YOLOv8 segmentation model
    model = YOLO('yolov8s-seg.yaml')
    
    # Train the model
    results = model.train(
        data=os.path.join(data_dir, "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=8,
        name='pothole_detection'
    )
    
    return results

def evaluate_model(model_path):
    """Evaluate the trained model"""
    model = YOLO(model_path)
    metrics = model.val()
    return metrics

def run_inference(model_path, image_dir, num_images=5):
    """Run inference on sample images"""
    model = YOLO(model_path)
    images = os.listdir(image_dir)[:num_images]
    
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        results = model.predict(img_path)
        
        # Plot results
        for result in results:
            plotted_img = result.plot()
            plt.figure(figsize=(10,9))
            plt.imshow(plotted_img)
            plt.axis('off')
            plt.title(img_name)
            plt.show()

def main():
    """Main function to run the training pipeline"""
    try:
        # Setup and verify data
        data_dir = setup_training_data()
        
        # Analyze dataset
        analyze_dataset(data_dir)
        
        # Train model
        results = train_model(data_dir)
        
        # Evaluate model
        metrics = evaluate_model('runs/segment/pothole_detection/weights/best.pt')
        
        # Run inference on some validation images
        run_inference(
            'runs/segment/pothole_detection/weights/best.pt',
            os.path.join(data_dir, "valid", "images")
        )
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
