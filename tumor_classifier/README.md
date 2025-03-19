# Tumor Classification Web Application

A modern web application for classifying brain tumor types using a deep learning model. The application features a beautiful UI with drag-and-drop functionality and real-time results.

## Features

- Modern, responsive UI with Material-UI components
- Drag-and-drop image upload
- Real-time image preview
- Secure file handling with temporary storage
- FastAPI backend with TensorFlow model integration
- Real-time classification results with confidence scores

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your trained model weights file (`model_weights.h5`) in the backend directory.

5. Start the backend server:
```bash
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Drag and drop an MRI image or click to select one
3. Click "Analyze Image" to get the classification results
4. View the predicted tumor type and confidence score

## API Endpoints

- `POST /predict`: Upload an image for classification
- `GET /health`: Check the health status of the API

## Security Features

- Images are processed in memory and not stored permanently
- CORS protection enabled
- Input validation and error handling
- Secure file type validation

## Technologies Used

- Frontend:
  - React with TypeScript
  - Material-UI
  - React Dropzone
  - Framer Motion
  - Axios

- Backend:
  - FastAPI
  - TensorFlow
  - Python-Multipart
  - Pillow
  - Uvicorn 