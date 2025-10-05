# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project that classifies handwriting images as either human-written or computer-generated using a Convolutional Neural Network (CNN). The project consists of:

1. **Data Generation**: Creates synthetic computer-generated text images using various fonts from the `FONTS/` directory
2. **Model Training**: Uses TensorFlow/Keras to train a CNN classifier (implemented in Jupyter notebook)
3. **Web Application**: Flask app that serves predictions via a web interface

## Development Commands

### Running the Web Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask development server
python app.py
# OR set environment variable and use flask command
export FLASK_APP=app.py
flask run
```

The web application will be available at `http://127.0.0.1:5000`

### Model Training and Development

```bash
# Start Jupyter notebook for model development
jupyter notebook
# Then open: "Human Vs Computer Handwriting Classification Project Final Khushboo.ipynb"
```

### Testing and Validation

- Upload test images through the web interface
- Images are automatically preprocessed (resized to 224x224, normalized)
- Model outputs binary classification: Human-generated vs Computer-generated

## Architecture

### Core Files

- **`app.py`**: Flask web application with file upload, image preprocessing, and model inference
- **`Word_Prediction.keras`**: Pre-trained CNN model (loaded at startup)
- **`templates/index.html`**: Bootstrap-based web interface with file upload and result display
- **`requirements.txt`**: Python dependencies (Flask, TensorFlow, NumPy, Pillow, etc.)

### Data Structure

```
├── FONTS/              # Font files (.ttf, .otf) for generating synthetic text
├── hand_words/         # Human handwriting dataset (PNG images)
├── comp_words/         # Computer-generated word images (JPG images)
├── data_split/         # Preprocessed and split datasets
│   ├── train/          # Training data
│   ├── valid/          # Validation data
│   └── test/           # Test data
├── uploads/            # User uploaded images (created at runtime)
└── templates/          # Flask HTML templates
```

### Model Architecture

- Input: 224x224 RGB images
- CNN architecture defined in Jupyter notebook
- Binary classification output (sigmoid activation)
- Trained on balanced dataset of human vs computer-generated text images

### Image Preprocessing Pipeline

1. Load image using Keras `image.load_img()` with target size (224, 224)
2. Convert to array and add batch dimension
3. Normalize pixel values to [0, 1] range
4. Pass through trained model for prediction

## Key Technical Details

- **Framework**: Flask for web app, TensorFlow/Keras for ML
- **Image Processing**: PIL/Pillow, OpenCV for data preprocessing
- **Model Format**: Keras .keras format (compiled with Adam optimizer, binary crossentropy loss)
- **Security**: Uses `secure_filename()` and UUID for uploaded files
- **File Handling**: Automatic creation of upload directory, temporary file storage

## Dataset Information

- **Human Handwriting**: IAM Handwriting Database format (based on file naming patterns)
- **Computer Generated**: Synthetic images created using various fonts from FONTS directory
- **Preprocessing**: Images are preprocessed and stored in data_split directories
- **Classes**: Binary classification (0 = Computer-generated, 1 = Human-generated)

## Development Notes

- Model requires recompilation after loading (`model.compile()` in app.py)
- Web interface uses Bootstrap 5.3.2 for styling
- File uploads are secured and given unique filenames
- Background image uses Unsplash CDN
- Prediction threshold: >0.5 for human-generated classification