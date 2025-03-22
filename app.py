import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model_path = '/Users/shreysharma/Desktop/improved_skin_detection.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

IMG_SIZE = 128

def process_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image) / 255.0 
        image_array = np.expand_dims(image_array, axis=0)  
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

#@app.route("/", methods=["GET"])
#def home():
    #return jsonify({"message": "Welcome to Skin Disease Prediction API"}), 200
@app.route('/')
def home():
    return render_template('index.html') 


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    processed_image = process_image(image_bytes)
    if processed_image is None:
        return jsonify({"error": "Image processing failed"}), 500

    predictions = model.predict(processed_image)
    predicted_class = int(np.argmax(predictions))  

    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] 
    predicted_disease = class_names[predicted_class]

    return jsonify({"prediction": predicted_disease}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
