
import os
import uuid
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'handwriting-classifier-secret-key'

# Load the trained model
try:
    model = load_model('Word_Prediction.keras', compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(file_path):
    """Load and preprocess the image for prediction."""
    if model is None:
        raise Exception("Model not loaded. Please check if Word_Prediction.keras exists.")
    
    try:
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize pixel values

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        confidence = float(predictions[0][0])

        # Return detailed prediction results
        if confidence > 0.5:
            prediction = "Human Generated"
            probability = confidence
        else:
            prediction = "Computer Generated"
            probability = 1 - confidence
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability': probability * 100,  # Convert to percentage
            'raw_score': confidence
        }
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(f"POST request received")
        print(f"Request files: {request.files}")
        print(f"Request form: {request.form}")
        
        # Check if the post request has the file part
        if 'file' not in request.files:
            print("No 'file' key in request.files")
            return render_template('index.html', error='No file part in the request. Please try again.')
        
        file = request.files['file']
        print(f"File object: {file}")
        print(f"File filename: {file.filename}")
        
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print("Empty filename")
            return render_template('index.html', error='No file selected. Please choose an image file.')
        
        if file and file.filename:
            try:
                # Check if it's a valid image file
                allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
                original_filename = secure_filename(file.filename)
                extension = os.path.splitext(original_filename)[1].lower()
                
                if extension not in allowed_extensions:
                    return render_template('index.html', error=f'Invalid file type. Please upload an image file ({", ".join(allowed_extensions)})')
                
                # Create unique filename
                filename = str(uuid.uuid4()) + extension
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                print(f"Saving file to: {file_path}")
                file.save(file_path)
                
                # Verify file was saved
                if not os.path.exists(file_path):
                    return render_template('index.html', error='Failed to save uploaded file. Please try again.')
                
                print(f"File saved successfully, making prediction...")
                result = predict_image(file_path)
                print(f"Prediction result: {result}")
                
                return render_template('index.html', result=result, filename=filename)
                
            except Exception as e:
                print(f"Error processing file: {str(e)}")
                return render_template('index.html', error=f'Error processing file: {str(e)}')
        else:
            return render_template('index.html', error='Invalid file. Please select a valid image file.')
    
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def simple_test():
    """Simple test route without complex JavaScript"""
    if request.method == 'POST':
        print(f"Simple test POST request received")
        print(f"Request files: {request.files}")
        
        if 'file' not in request.files:
            return render_template('simple_test.html', error='No file part in the request.')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('simple_test.html', error='No file selected.')
        
        if file and file.filename:
            try:
                original_filename = secure_filename(file.filename)
                extension = os.path.splitext(original_filename)[1].lower()
                filename = str(uuid.uuid4()) + extension
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                file.save(file_path)
                result = predict_image(file_path)
                return render_template('simple_test.html', result=result, filename=filename)
                
            except Exception as e:
                return render_template('simple_test.html', error=f'Error: {str(e)}')
    
    return render_template('simple_test.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
