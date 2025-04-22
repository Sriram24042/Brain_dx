# BrainDx - AI-Powered Brain Tumor Classification

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/React-19.0.0-61DAFB.svg" alt="React Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg" alt="TensorFlow Version">
</div>

## 🧠 Revolutionizing Brain Tumor Diagnosis with AI

BrainDx is a cutting-edge web application that leverages advanced deep learning to classify brain tumors from MRI scans with remarkable accuracy. By combining state-of-the-art machine learning techniques with an intuitive user interface, BrainDx aims to assist healthcare professionals in making faster, more accurate diagnostic decisions.


### ✨ Key Features

- **Real-time Classification**: Upload MRI scans and receive instant tumor classification results
- **High Accuracy**: Achieves 97% classification accuracy across multiple tumor types
- **Modern UI**: Beautiful, responsive interface with smooth animations
- **Secure Processing**: All image processing happens locally or in secure cloud environments
- **Comprehensive Analysis**: Classifies between No Tumor, Meningioma, Glioma, and Pituitary Tumor

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- Git

### Detailed Installation Guide

#### 1. Clone the Repository

```bash
git clone https://github.com/Sriram24042/Brain_dx.git
cd Brain_dx
```

#### 2. Backend Setup

```bash
# Navigate to the backend directory
cd tumor_classifier/backend

# Create and activate a virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will automatically download the pre-trained model on first run if it's not already present.

#### 3. Frontend Setup

```bash
# Open a new terminal and navigate to the frontend directory
cd tumor_classifier/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173` by default.

#### 4. Access the Application

Open your browser and navigate to `http://localhost:5173` to use the application.

## 🔧 Project Structure

```
Brain_dx/
├── tumor_classifier/
│   ├── backend/                # FastAPI server
│   │   ├── main.py             # Main application file
│   │   ├── requirements.txt    # Python dependencies
│   │   └── brain_tumor_classification_model.h5  # Pre-trained model
│   └── frontend/               # React application
│       ├── src/                # Source code
│       ├── public/             # Static assets
│       └── package.json        # Node.js dependencies
```
## Note : Download The Trained Model Weights From The Google Drive Link Provided Above And Store It According To The File Structure 
## 🧪 How It Works

1. **Image Upload**: Users upload brain MRI scans through the intuitive drag-and-drop interface
2. **Preprocessing**: Images are automatically resized and normalized
3. **Classification**: The deep learning model analyzes the image and identifies tumor types
4. **Results Display**: Results are presented with confidence scores and visual feedback

## 🔍 API Documentation

### Endpoints

- `POST /predict`: Upload and classify brain MRI images
  - Accepts: Multipart form data with image file
  - Returns: JSON with classification results and confidence scores

- `GET /health`: Health check endpoint
  - Returns: Server status and model loading information

## 🛠️ Tech Stack

### Frontend
- **React 19** with TypeScript for type safety
- **Material-UI** for beautiful, responsive components
- **Framer Motion** for smooth animations and transitions
- **Axios** for API communication

### Backend
- **FastAPI** for high-performance API endpoints
- **TensorFlow/Keras** for deep learning model
- **Python 3.8+** for backend logic
- **Scikit-learn** for data preprocessing


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact me at sriram24042@gmail.com.

---

<div align="center">
  <p>Made with ❤️ by Sriram24042</p>
</div> 
