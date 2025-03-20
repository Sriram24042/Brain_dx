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
            try:
                # First attempt: Try loading with custom_objects
                custom_objects = {
                    'InputLayer': tf.keras.layers.InputLayer
                }
                model = tf.keras.models.load_model(path, custom_objects=custom_objects)
            except Exception as e1:
                print(f"First attempt failed: {str(e1)}")
                try:
                    # Second attempt: Try loading with compile=False
                    model = tf.keras.models.load_model(path, compile=False)
                except Exception as e2:
                    print(f"Second attempt failed: {str(e2)}")
                    try:
                        # Third attempt: Try loading weights only
                        base_model = tf.keras.applications.MobileNetV2(
                            input_shape=(224, 224, 3),
                            include_top=False,
                            weights='imagenet'
                        )
                        x = base_model.output
                        x = tf.keras.layers.GlobalAveragePooling2D()(x)
                        x = tf.keras.layers.Dense(4, activation='softmax')(x)
                        model = tf.keras.Model(inputs=base_model.input, outputs=x)
                        model.load_weights(path)
                    except Exception as e3:
                        print(f"Third attempt failed: {str(e3)}")
                        raise Exception("All model loading attempts failed")
            
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