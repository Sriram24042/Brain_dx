from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import tempfile
import shutil
import gdown

app = FastAPI(title="BrainDx API")

# Configure CORS
allowed_origins = [
    "http://localhost:5173",  # Local development
    "https://brain-dx-1.vercel.app",  # Your Vercel domain
    "https://brain-dx.vercel.app",    # Alternative Vercel domain
    "https://braindx.vercel.app",     # Another alternative
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    print("Attempting to load model...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    
    # Google Drive file ID from your link
    file_id = "1YAr1sX2D92BZ41DFuiUgC8p4opsSIWNw"
    output_path = "model_weights.h5"
    
    # Download the model if it doesn't exist
    if not os.path.exists(output_path):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print("Model downloaded successfully!")
    
    # Try loading the model
    custom_objects = {
        'InputLayer': tf.keras.layers.InputLayer,
        'Conv2D': tf.keras.layers.Conv2D,
        'MaxPooling2D': tf.keras.layers.MaxPooling2D,
        'Flatten': tf.keras.layers.Flatten,
        'Dense': tf.keras.layers.Dense,
        'Dropout': tf.keras.layers.Dropout,
        'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
        'GlobalAveragePooling3D': tf.keras.layers.GlobalAveragePooling3D,
        'Concatenate': tf.keras.layers.Concatenate,
        'MultiHeadAttention': tf.keras.layers.MultiHeadAttention,
        'LayerNormalization': tf.keras.layers.LayerNormalization,
        'Add': tf.keras.layers.Add,
        'Reshape': tf.keras.layers.Reshape,
        'Conv3D': tf.keras.layers.Conv3D
    }
    
    model = tf.keras.models.load_model(output_path, custom_objects=custom_objects, compile=False)
    CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Error type: {type(e)}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model path: {output_path}")
    if os.path.exists(output_path):
        print(f"Model file size: {os.path.getsize(output_path)} bytes")
    raise Exception("Failed to load model weights")

def preprocess_image(image):
    # Convert image to RGB mode
    image = image.convert('RGB')
    # Resize image to match model input size
    image = image.resize((224, 224))
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return JSONResponse({
            "class": predicted_class,
            "confidence": confidence,
            "message": f"Predicted tumor type: {predicted_class} with {confidence:.2%} confidence"
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 