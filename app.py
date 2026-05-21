# ============================================================
# Plant Disease Detection System - Flask Backend
# ============================================================
# This is the main server file. It handles:
# 1. Serving the web page
# 2. Receiving uploaded images
# 3. Running the ML model for prediction
# 4. Returning results to the browser
# ============================================================

import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

# Initialize the Flask app
app = Flask(__name__)

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
UPLOAD_FOLDER = 'static/uploads'          # Where uploaded images are saved
MODEL_PATH = 'model/plant_disease_model.h5'  # Path to trained model
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Accepted image formats
IMAGE_SIZE = (128, 128)                   # Model expects 128x128 images

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload

# ----------------------------------------------------------
# Disease Information Database
# This dictionary holds information about each disease class.
# Add or edit entries here to update what the app shows.
# ----------------------------------------------------------
DISEASE_INFO = {
    "Healthy": {
        "description": "The plant leaf appears to be in excellent health! No signs of disease or infection were detected.",
        "tips": [
            "Continue regular watering schedule",
            "Ensure adequate sunlight exposure",
            "Apply balanced fertilizer monthly",
            "Monitor for early signs of pests"
        ],
        "severity": "none",
        "color": "#22c55e"
    },
    "Early Blight": {
        "description": "Early Blight is a fungal disease caused by Alternaria solani. It causes dark, concentric spots on older leaves, often surrounded by a yellow halo.",
        "tips": [
            "Remove and destroy infected leaves immediately",
            "Apply copper-based fungicide every 7–10 days",
            "Avoid overhead watering to reduce humidity",
            "Rotate crops each season to prevent recurrence",
            "Ensure proper plant spacing for airflow"
        ],
        "severity": "moderate",
        "color": "#f59e0b"
    },
    "Late Blight": {
        "description": "Late Blight is caused by Phytophthora infestans. It spreads rapidly in wet, cool conditions and can destroy an entire crop within days.",
        "tips": [
            "Apply systemic fungicide (e.g., Mancozeb) immediately",
            "Remove heavily infected plants to prevent spread",
            "Avoid working in field when plants are wet",
            "Use disease-resistant plant varieties next season",
            "Improve drainage to reduce soil moisture"
        ],
        "severity": "high",
        "color": "#ef4444"
    },
    "Leaf Mold": {
        "description": "Leaf Mold is caused by the fungus Passalora fulva. It appears as pale greenish-yellow spots on upper leaf surfaces with olive-colored mold underneath.",
        "tips": [
            "Reduce humidity by improving greenhouse ventilation",
            "Apply fungicide containing chlorothalonil",
            "Remove and dispose of infected leaves carefully",
            "Space plants wider to improve air circulation",
            "Avoid wetting foliage during irrigation"
        ],
        "severity": "moderate",
        "color": "#f97316"
    }
}

# Class names must match the order used during model training
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight", "Leaf Mold"]

# ----------------------------------------------------------
# Load the trained model
# ----------------------------------------------------------
model = None

def load_model():
    """Load the Keras model from disk (called once at startup)."""
    global model
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Model loaded successfully!")
    else:
        print(f"[WARNING] Model file not found at {MODEL_PATH}")
        print("[WARNING] Run 'python train_model.py' first to train and save the model.")

# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

def allowed_file(filename):
    """Check if the uploaded file has an accepted extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Load and prepare the image for model prediction.
    Steps:
    1. Open the image
    2. Convert to RGB (removes alpha channel if present)
    3. Resize to model's expected input size
    4. Normalize pixel values from [0,255] to [0,1]
    5. Add batch dimension (model expects batches, not single images)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0          # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
    return img_array


def predict_disease(image_path):
    """
    Run the model on the preprocessed image and return results.
    Returns a dict with class name, confidence, and disease info.
    """
    if model is None:
        # If model isn't loaded, return a demo response for testing
        return get_demo_prediction()

    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)           # Get probability scores
    predicted_index = np.argmax(predictions[0])      # Index of highest score
    confidence = float(predictions[0][predicted_index]) * 100  # As percentage

    disease_name = CLASS_NAMES[predicted_index]
    disease_data = DISEASE_INFO.get(disease_name, {})

    return {
        "disease": disease_name,
        "confidence": round(confidence, 2),
        "description": disease_data.get("description", ""),
        "tips": disease_data.get("tips", []),
        "severity": disease_data.get("severity", "unknown"),
        "color": disease_data.get("color", "#6b7280"),
        "all_scores": {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }


def get_demo_prediction():
    """
    Returns a demo prediction when no model is loaded.
    Useful for testing the frontend without a trained model.
    """
    import random
    disease_name = random.choice(CLASS_NAMES)
    confidence = round(random.uniform(75, 98), 2)
    disease_data = DISEASE_INFO[disease_name]
    return {
        "disease": disease_name,
        "confidence": confidence,
        "description": disease_data["description"],
        "tips": disease_data["tips"],
        "severity": disease_data["severity"],
        "color": disease_data["color"],
        "demo_mode": True,
        "all_scores": {name: round(random.uniform(1, 20), 2) for name in CLASS_NAMES}
    }


# ----------------------------------------------------------
# Routes (URL endpoints)
# ----------------------------------------------------------

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return prediction.
    Expects a multipart/form-data POST with an 'image' field.
    """
    # Check if file was included in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # Check if user actually selected a file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check file extension is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG'}), 400

    # Save the uploaded file
    filename = 'uploaded_leaf.jpg'
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Run prediction
    result = predict_disease(save_path)

    # Add image URL so frontend can display it
    result['image_url'] = f'/static/uploads/{filename}'

    return jsonify(result)


# ----------------------------------------------------------
# Run the app
# ----------------------------------------------------------
if __name__ == '__main__':
    # Make sure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Load the trained model
    load_model()

    # Start Flask development server
    print("\n[INFO] Starting Plant Disease Detection Server...")
    print("[INFO] Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
