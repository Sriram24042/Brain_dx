from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)

# Allow CORS from all origins (for development purposes only)
CORS(app)

# Configuration
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_classification_model.h5")
model = load_model(MODEL_PATH)

# Class indices mapping
class_indices = {'No Tumor': 0, 'Meningioma': 1, 'Glioma': 2, 'Pituitary Tumor': 3}
reverse_class_indices = {v: k for k, v in class_indices.items()}

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Run prediction
    try:
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = reverse_class_indices[predicted_class_index]
        confidence = float(np.max(predictions))

        # Check if the prediction is a valid brain tumor class
        if predicted_class == 'No Tumor':
            os.remove(filepath)
            return jsonify({"error": "The uploaded image is not a brain tumor MRI."}), 400

        # Clean up uploaded file after processing
        os.remove(filepath)

        return jsonify({"class": predicted_class, "confidence": confidence})

    except Exception as e:
        os.remove(filepath)
        return jsonify({"error": "Error in processing the image."}), 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Remove debug=True
