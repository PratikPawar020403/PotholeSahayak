# Pothole Detection System 🛣️

A deep learning-powered system that detects potholes in images and videos using YOLOv8 technology. This application is designed to assist in road maintenance and infrastructure monitoring.

## Features 🌟

- **Fast Detection**: Real-time pothole detection in images and videos
- **High Accuracy**: Utilizing state-of-the-art YOLOv8 model
- **Easy to Use**: Simple upload and analyze interface
- **Mobile Responsive**: Works well on both desktop and mobile devices

## Installation 🔧

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the application:
   - Load the model
   - Upload an image or video
   - View detection results

## Model Details 🧠

- Architecture: YOLOv8
- Input Resolution: 640x640
- Output: Pothole locations with bounding boxes
- Performance: Fast inference with high accuracy

## Project Structure 📁

```
pothole-detection/
├── app.py              # Main Streamlit application
├── train.py           # Training script
├── requirements.txt   # Project dependencies
├── best.pt           # Trained model
├── media/            # Sample media files
```

## Training 🎯

To train the model on your own dataset:

1. Prepare your dataset in YOLO format
2. Update configuration in `train.py`
3. Run training:
```bash
python train.py
train2.py is google colab compactiable code . 
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments 🙏

- YOLOv8 by Ultralytics
- Streamlit for the web interface
