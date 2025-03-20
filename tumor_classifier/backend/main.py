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
import uvicorn
import hashlib

app = FastAPI(title="BrainDx API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Model configuration
MODEL_PATH = "model_weights.h5"
GOOGLE_DRIVE_ID = "1HVVWWxcDgCWjvZDXoNG7s0E8PsTyE8g_"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

def verify_model_file(file_path):
    """Verify that the model file is valid."""
    try:
        # Check if file exists and has content
        if not os.path.exists(file_path):
            print(f"Model file not found at: {file_path}")
            return False
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"Model file is empty: {file_path}")
            return False
        
        # Try to load the model to verify it's valid
        try:
            tf.keras.models.load_model(file_path, compile=False)
            print(f"Model file verified successfully. Size: {file_size / (1024*1024):.2f} MB")
            return True
        except Exception as e:
            print(f"Error verifying model file: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error checking model file: {str(e)}")
        return False

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive using gdown."""
    print(f"Attempting to download model from Google Drive (ID: {file_id})...")
    
    try:
        # First try with gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Attempting download with gdown...")
        
        # Use gdown with specific options for large files
        success = gdown.download(
            url=url,
            output=destination,
            quiet=False,
            fuzzy=True,
            use_cookies=False,
            verify=True  # Enable SSL verification
        )
        
        if not success:
            print("Failed to download with gdown, trying alternative method...")
            
            # Alternative method using direct download with cookies
            session = requests.Session()
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # First request to get the cookie
            response = session.get(url, verify=True)  # Enable SSL verification
            
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
            response = session.get(url, stream=True, verify=True)  # Enable SSL verification
            
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                return False
            
            # Get file size from headers
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                print("Error: Could not determine file size")
                return False
                
            print(f"Total file size: {total_size / (1024*1024):.2f} MB")
            
            # Download with progress bar
            with open(destination, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):  # Increased chunk size
                    size = f.write(data)
                    pbar.update(size)
        
        print(f"Download completed. Verifying file...")
        
        # Verify the downloaded file
        if verify_model_file(destination):
            print(f"File verified and saved to: {destination}")
            return True
        else:
            print("File verification failed. Download was unsuccessful.")
            if os.path.exists(destination):
                os.remove(destination)
            return False
            
    except Exception as e:
        print(f"Error during download: {str(e)}")
        if os.path.exists(destination):
            os.remove(destination)
        return False

# Load model at startup
print("Attempting to load model...")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

try:
    if not os.path.exists(MODEL_PATH) or not verify_model_file(MODEL_PATH):
        print("Model file not found or invalid. Downloading from Google Drive...")
        if download_file_from_google_drive(GOOGLE_DRIVE_ID, MODEL_PATH):
            print("Model downloaded and verified successfully")
        else:
            raise Exception("Failed to download and verify model")
    
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
            'BatchNormalization': tf.keras.layers.BatchNormalization,
            'input_layer': tf.keras.layers.InputLayer,
            'input_1': tf.keras.layers.InputLayer,
            'input': tf.keras.layers.InputLayer
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 