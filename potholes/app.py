import streamlit as st
import cv2
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from PIL import Image
import time
import io
import glob
from datetime import datetime
import re
import json
from pathlib import Path
import base64
import torch

# Set page configuration
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define application structure
class PotholeDetectionApp:
    def __init__(self) -> None:
        # Initialize session state variables if they don't exist
        if 'model' not in st.session_state:
            st.session_state.model = None
            
        # Define pages
        self.pages: Dict[str, callable] = {
            "🏠 Home": self.home_page,
            "🔍 Inference": self.inference_page,
        }
        
        # Add custom styling for better mobile responsiveness
        self.add_custom_css()
        
    def add_custom_css(self) -> None:
        """Add custom CSS for modern styling and mobile responsiveness"""
        st.markdown("""
            <style>
            /* Base styles */
            .stApp {
                background-color: #F0EAE4;
            }
            
            /* Responsive container */
            .css-1d391kg {
                padding: 1rem;
            }
            
            /* Button styling */
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 20px;
                padding: 0.5rem 2rem;
                border: none;
                transition: all 0.3s ease;
                width: 100%;
                max-width: 300px;
                margin: 0 auto;
            }
            
            /* Metric styling */
            .stMetric {
                background-color: white;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
            }
            
            /* Alert styling */
            .stAlert {
                background-color: white;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
            }
            
            /* Responsive columns */
            @media (max-width: 768px) {
                .row-widget.stHorizontal {
                    flex-direction: column;
                }
                .row-widget.stHorizontal > div {
                    width: 100% !important;
                    margin-bottom: 1rem;
                }
                /* Adjust image size on mobile */
                .stImage > img {
                    max-width: 100% !important;
                    height: auto !important;
                }
                /* Make video responsive */
                .stVideo > video {
                    width: 100% !important;
                    height: auto !important;
                }
            }
            
            /* Improve text readability on mobile */
            @media (max-width: 480px) {
                p, li {
                    font-size: 16px !important;
                    line-height: 1.6 !important;
                }
                h1 { font-size: 24px !important; }
                h2 { font-size: 20px !important; }
                h3 { font-size: 18px !important; }
            }
            </style>
        """, unsafe_allow_html=True)
    
    def run(self) -> None:
        """Run the application"""
        # Sidebar
        with st.sidebar:
            st.title("Navigation")
            selection = st.radio("Go to", list(self.pages.keys()))
        
        # Display selected page
        self.pages[selection]()
    
    def load_model(self) -> Optional[YOLO]:
        """Load the YOLOv8 model"""
        try:
            model = YOLO('potholes/best.pt')
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def home_page(self) -> None:
        """Display the home page"""
        st.title("🛣️ Pothole Detection System")
        
        # Introduction in a clean, minimal style
        st.markdown("""
        ### About
        A deep learning-powered system that detects potholes in images and videos using YOLOv8 technology. 
        Designed to assist in road maintenance and infrastructure monitoring.
        
        ### Features
        - **Fast Detection**: Real-time pothole detection in images and videos
        - **High Accuracy**: Utilizing state-of-the-art YOLOv8 model
        - **Easy to Use**: Simple upload and analyze interface
        """)
        
        # Responsive two column layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Steps to Use")
            st.markdown("""
            1. Load the model
            2. Upload image or video
            3. View detection results
            """)
            
            st.markdown("### Model Details")
            st.markdown("""
            - Architecture: YOLOv8
            - Input: Images/Videos
            - Output: Pothole locations with bounding boxes
            - Resolution: 640x640
            - Performance: Fast inference with high accuracy
            """)
            
            st.markdown("### Tips for Best Results")
            st.markdown("""
            - Upload clear, well-lit images/videos
            - Supported formats: JPG, PNG, MP4
            - For videos: Stable footage works best
            - Optimal resolution: 640x640 or higher
            """)
        
        with col2:
            st.markdown("### Model Information")
            st.markdown("""
            The model used in this application is a YOLOv8 Segmentation model, which is a state-of-the-art deep learning model for object detection tasks. 
            It is trained on a large dataset of images and videos of roads and potholes, and is able to detect potholes with high accuracy.
            """)
    
    def process_image(self, image: np.ndarray, model: YOLO) -> Tuple[np.ndarray, List[dict]]:
        """Process an image and return the results"""
        results = model.predict(image, conf=0.25)
        plotted_img = results[0].plot()
        return plotted_img, results[0].boxes.data.tolist()
    
    def process_video(self, video_path: str, model: YOLO) -> str:
        """Process a video and return the output path"""
        try:
            output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
            results = model.predict(video_path, save=True, project=tempfile.gettempdir())
            return output_path
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return ""
    
    def inference_page(self) -> None:
        """Display the inference page for image/video upload and processing"""
        st.title("🔍 Inference")
        
        # Load model button
        if st.button("Load Model"):
            st.session_state.model = self.load_model()
        
        if st.session_state.model is None:
            st.warning("Please load the model first!")
            return
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image or video file", type=['jpg', 'jpeg', 'png', 'mp4'])
        
        if uploaded_file is not None:
            file_type = uploaded_file.type.split('/')[0]
            
            try:
                if file_type == "image":
                    self.handle_image_upload(uploaded_file)
                elif file_type == "video":
                    self.handle_video_upload(uploaded_file)
                else:
                    st.error("Unsupported file type")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    def handle_image_upload(self, uploaded_file: io.BytesIO) -> None:
        """Handle image upload and processing"""
        # Create columns for original and processed images
        col1, col2 = st.columns(2)
        
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Show original image
        with col1:
            st.markdown("### Original Image")
            st.image(image, channels="RGB", use_column_width=True)
        
        # Process and show results
        with col2:
            st.markdown("### Processed Image")
            processed_img, detections = self.process_image(image, st.session_state.model)
            st.image(processed_img, channels="RGB", use_column_width=True)
            
            if len(detections) > 0:
                st.success(f"Found {len(detections)} potholes!")
            else:
                st.info("No potholes detected.")
    
    def handle_video_upload(self, uploaded_file: io.BytesIO) -> None:
        """Handle video upload and processing"""
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Process video
        st.markdown("### Processing Video")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            output_path = self.process_video(tfile.name, st.session_state.model)
            if output_path and os.path.exists(output_path):
                st.markdown("### Results")
                st.video(output_path)
            else:
                st.error("Error processing video")
        finally:
            # Cleanup
            os.unlink(tfile.name)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)

# Run the application
if __name__ == "__main__":
    app = PotholeDetectionApp()
    app.run()
