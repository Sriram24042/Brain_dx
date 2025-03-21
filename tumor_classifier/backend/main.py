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
MODEL_PATH = "brain_tumor_classification_model.h5"  # Match the local filename
GOOGLE_DRIVE_ID = "1HVVWWxcDgCWjvZDXoNG7s0E8PsTyE8g_"

# Class indices mapping (matching local implementation)
CLASS_INDICES = {'No Tumor': 0, 'Meningioma': 1, 'Glioma': 2, 'Pituitary Tumor': 3}
REVERSE_CLASS_INDICES = {v: k for k, v in CLASS_INDICES.items()}

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
        print("Attempting download with gdown...")
        success = gdown.download(
            url=f"https://drive.google.com/uc?id={file_id}",
            output=destination,
            quiet=False,
            fuzzy=True,
            use_cookies=True
        )
        
        if not success:
            print("gdown download failed, trying alternative method...")
            
            # Alternative method using direct download
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            print(f"Attempting direct download from: {url}")
            
            response = requests.get(url, stream=True)
            
            if response.status_code != 200:
                print(f"Direct download failed with status code {response.status_code}")
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
                for data in response.iter_content(chunk_size=8192):
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
        print(f"Error type: {type(e)}")
        if os.path.exists(destination):
            os.remove(destination)
        return False

def load_model():
    """Load the model from file or download if needed."""
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    
    # First try to load the model directly
    try:
        print(f"Attempting to load model from: {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            print(f"Model file exists. Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("Model loaded successfully!")
                return model
            except Exception as e:
                print(f"Error loading model directly: {str(e)}")
        else:
            print("Model file not found locally")
    except Exception as e:
        print(f"Error in initial load attempt: {str(e)}")
    
    # If direct load fails, try downloading from Google Drive
    print("Attempting to download model from Google Drive...")
    try:
        # Download using gdown
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        print(f"Downloading from: {url}")
        
        success = gdown.download(url, MODEL_PATH, quiet=False)
        if not success:
            print("gdown download failed")
            return None
            
        print(f"Download completed. File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        
        # Try loading the downloaded model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully after download!")
        return model
        
    except Exception as e:
        print(f"Error during download/load: {str(e)}")
        print(f"Error type: {type(e)}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return None

# Load model at startup
print("Starting model loading process...")
model = load_model()

if model is None:
    raise Exception("Failed to load model")

print("Model initialization complete!")

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
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = REVERSE_CLASS_INDICES[predicted_class_index]
        confidence = float(np.max(predictions[0]))
        
        # Check if the prediction is a valid brain tumor class
        if predicted_class == 'No Tumor':
            return JSONResponse({
                "error": "The uploaded image is not a brain tumor MRI."
            }, status_code=400)
        
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