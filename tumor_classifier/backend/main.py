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
import requests
from tqdm import tqdm

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

# Model configuration
MODEL_PATH = "model_weights.h5"
GOOGLE_DRIVE_ID = "1YAr1sX2D92BZ41DFuiUgC8p4opsSIWNw"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive using direct download URL."""
    print(f"Attempting to download model from Google Drive (ID: {file_id})...")
    
    # Create direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # First request to get the cookie
    session = requests.Session()
    response = session.get(url)
    
    # Get the token from the cookie
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    # If we got a token, add it to the URL
    if token:
        url = f"{url}&confirm={token}"
    
    # Download the file
    response = session.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    print(f"Total file size: {total_size / (1024*1024):.2f} MB")
    
    with open(destination, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    print(f"Download completed. File saved to: {destination}")
    return True

# Load model at startup
print("Attempting to load model...")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

try:
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Downloading from Google Drive...")
        if download_file_from_google_drive(GOOGLE_DRIVE_ID, MODEL_PATH):
            print("Model downloaded successfully")
        else:
            raise Exception("Failed to download model")
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model path: {MODEL_PATH}")
    print(f"File exists: {os.path.exists(MODEL_PATH)}")
    if os.path.exists(MODEL_PATH):
        print(f"File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    
    # Load the model with custom objects
    model = tf.keras.models.load_model(MODEL_PATH, 
        custom_objects={
            'Dense': tf.keras.layers.Dense,
            'Conv2D': tf.keras.layers.Conv2D,
            'MaxPooling2D': tf.keras.layers.MaxPooling2D,
            'Flatten': tf.keras.layers.Flatten,
            'Dropout': tf.keras.layers.Dropout,
            'InputLayer': tf.keras.layers.InputLayer,
            'Input': tf.keras.layers.Input,
            'Model': tf.keras.Model,
            'Sequential': tf.keras.Sequential,
            'ReLU': tf.keras.layers.ReLU,
            'BatchNormalization': tf.keras.layers.BatchNormalization
        },
        compile=False
    )
    CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Error type: {type(e)}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model path: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        print(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")
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