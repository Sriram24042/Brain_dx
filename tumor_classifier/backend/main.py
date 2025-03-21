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
import requests
from tqdm import tqdm
import uvicorn
import hashlib
import sys
import base64

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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_classification_model.h5")
BACKUP_MODEL_URL = "https://huggingface.co/spaces/Sriram24824/BrainDx/resolve/main/brain_tumor_classification_model.h5"

# Class indices mapping (matching local implementation)
CLASS_INDICES = {'No Tumor': 0, 'Meningioma': 1, 'Glioma': 2, 'Pituitary Tumor': 3}
REVERSE_CLASS_INDICES = {v: k for k, v in CLASS_INDICES.items()}

def verify_model_file(file_path):
    """Verify that the model file is valid."""
    try:
        if not os.path.exists(file_path):
            print(f"Model file not found at: {file_path}")
            return False
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"Model file is empty: {file_path}")
            return False
        
        try:
            # Try loading with custom_objects to handle compatibility
            custom_objects = {
                'InputLayer': tf.keras.layers.InputLayer,
                'Conv2D': tf.keras.layers.Conv2D,
                'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                'Flatten': tf.keras.layers.Flatten,
                'Dense': tf.keras.layers.Dense,
                'Dropout': tf.keras.layers.Dropout
            }
            
            model = tf.keras.models.load_model(
                file_path,
                custom_objects=custom_objects,
                compile=False
            )
            
            # Verify model structure
            if len(model.layers) == 0:
                print("Model has no layers")
                return False
                
            # Verify input shape
            if model.input_shape != (None, 224, 224, 3):
                print(f"Unexpected input shape: {model.input_shape}")
                return False
                
            print(f"Model file verified successfully. Size: {file_size / (1024*1024):.2f} MB")
            print(f"Model summary:")
            model.summary()
            return True
            
        except Exception as e:
            print(f"Error verifying model file: {str(e)}")
            print(f"Attempting alternative loading method...")
            
            try:
                # Try reconstructing the model architecture
                inputs = tf.keras.Input(shape=(224, 224, 3))
                x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                x = tf.keras.layers.Flatten()(x)
                x = tf.keras.layers.Dense(64, activation='relu')(x)
                outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
                
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.load_weights(file_path)
                print("Successfully loaded model weights")
                model.summary()
                return True
                
            except Exception as e2:
                print(f"Error in alternative loading method: {str(e2)}")
                return False
            
    except Exception as e:
        print(f"Error checking model file: {str(e)}")
        return False

def download_file_with_progress(url, destination):
    """Download a file with progress bar and verification."""
    print(f"Downloading model from: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(destination, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        
        print(f"Download completed. File size: {os.path.getsize(destination) / (1024*1024):.2f} MB")
        return True
        
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
    print(f"Model path: {MODEL_PATH}")
    print(f"Model directory exists: {os.path.exists(os.path.dirname(MODEL_PATH))}")
    
    # First try to load the model directly
    try:
        print(f"Attempting to load model from: {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            print(f"Model file exists. Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
            if verify_model_file(MODEL_PATH):
                # Use the same loading method that worked in verify_model_file
                custom_objects = {
                    'InputLayer': tf.keras.layers.InputLayer,
                    'Conv2D': tf.keras.layers.Conv2D,
                    'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                    'Flatten': tf.keras.layers.Flatten,
                    'Dense': tf.keras.layers.Dense,
                    'Dropout': tf.keras.layers.Dropout
                }
                
                try:
                    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                except:
                    # If direct loading fails, try the alternative method
                    inputs = tf.keras.Input(shape=(224, 224, 3))
                    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
                    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                    x = tf.keras.layers.Flatten()(x)
                    x = tf.keras.layers.Dense(64, activation='relu')(x)
                    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
                    
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)
                    model.load_weights(MODEL_PATH)
                
                print("Model loaded successfully!")
                return model
            else:
                print("Model file verification failed")
        else:
            print("Model file not found locally")
    except Exception as e:
        print(f"Error in initial load attempt: {str(e)}")
        print(f"Error type: {type(e)}")
    
    # If direct load fails, try downloading from backup URL
    print("Attempting to download model from backup URL...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Download the model
        if not download_file_with_progress(BACKUP_MODEL_URL, MODEL_PATH):
            print("Failed to download model from backup URL")
            return None
            
        # Verify and load the downloaded model
        if verify_model_file(MODEL_PATH):
            # Use the same loading method that worked in verify_model_file
            custom_objects = {
                'InputLayer': tf.keras.layers.InputLayer,
                'Conv2D': tf.keras.layers.Conv2D,
                'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                'Flatten': tf.keras.layers.Flatten,
                'Dense': tf.keras.layers.Dense,
                'Dropout': tf.keras.layers.Dropout
            }
            
            try:
                model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
            except:
                # If direct loading fails, try the alternative method
                inputs = tf.keras.Input(shape=(224, 224, 3))
                x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                x = tf.keras.layers.Flatten()(x)
                x = tf.keras.layers.Dense(64, activation='relu')(x)
                outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
                
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.load_weights(MODEL_PATH)
            
            print("Model loaded successfully after download!")
            return model
        else:
            print("Downloaded model verification failed")
            return None
        
    except Exception as e:
        print(f"Error during download/load: {str(e)}")
        print(f"Error type: {type(e)}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return None

# Load model at startup
print("Starting model loading process...")
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
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