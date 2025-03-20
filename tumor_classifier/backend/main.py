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
    
    # Try different possible paths
    model_paths = [
        "model_weights.h5",
        "tumor_classifier/backend/model_weights.h5",
        "/app/model_weights.h5",
        "/app/tumor_classifier/backend/model_weights.h5"
    ]
    
    model_loaded = False
    for path in model_paths:
        print(f"Checking path: {path}")
        if os.path.exists(path):
            print(f"Found model at: {path}")
            model = tf.keras.models.load_model(path)
            CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
            print("Model loaded successfully!")
            model_loaded = True
            break
    
    if not model_loaded:
        raise FileNotFoundError("Model file not found in any of the expected locations")
        
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Error type: {type(e)}")
    print(f"TensorFlow version: {tf.__version__}")
    model = None

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