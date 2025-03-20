# Brain Tumor Classification Project - Development Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Local Development Setup](#local-development-setup)
5. [Backend Development](#backend-development)
6. [Frontend Development](#frontend-development)
7. [Model Integration](#model-integration)
8. [Deployment Guide](#deployment-guide)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)

## Project Overview
A web application for classifying brain tumors from MRI images using deep learning. The application provides a user-friendly interface for uploading MRI images and displays classification results with confidence scores.

## Technology Stack
- **Frontend**: React + TypeScript + Vite
- **Backend**: FastAPI (Python)
- **ML Framework**: TensorFlow
- **Deployment**: 
  - Frontend: Vercel
  - Backend: Render
  - Model Storage: Git LFS

## Project Structure
```
tumor_classifier/
├── frontend/                 # React frontend application
│   ├── src/                 # Source code
│   ├── public/              # Static files
│   └── package.json         # Frontend dependencies
└── backend/                 # FastAPI backend application
    ├── main.py             # Main application file
    ├── requirements.txt    # Python dependencies
    └── model_weights.h5    # Trained model file
```

## Local Development Setup

### Prerequisites
1. Node.js (v16 or higher)
2. Python (v3.8 or higher)
3. Git
4. Git LFS

### Backend Setup
1. Create and activate virtual environment:
   ```bash
   cd tumor_classifier/backend
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Install dependencies:
   ```bash
   cd tumor_classifier/frontend
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

## Backend Development

### API Endpoints
1. `POST /predict`
   - Accepts image file
   - Returns tumor classification and confidence

2. `GET /health`
   - Health check endpoint
   - Returns service status and model loading status

### Model Integration
- Model file: `model_weights.h5`
- Supported classes: glioma, meningioma, no_tumor, pituitary
- Input image size: 224x224
- Supported formats: JPEG, PNG

## Frontend Development

### Features
1. Drag-and-drop file upload
2. Image preview
3. Real-time classification
4. Confidence score display
5. Error handling
6. Responsive design

### Environment Variables
```
VITE_API_URL=http://localhost:8000  # Development
VITE_API_URL=https://braindx-api.onrender.com  # Production
```

## Deployment Guide

### Backend Deployment (Render)
1. Create Render account
2. Connect GitHub repository
3. Create new Web Service
4. Configure settings:
   - Name: `braindx-api`
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT`
   - Root Directory: `tumor_classifier/backend`
5. Add environment variables:
   - `PORT`: (auto-set by Render)

### Frontend Deployment (Vercel)
1. Create Vercel account
2. Connect GitHub repository
3. Configure project:
   - Framework Preset: Vite
   - Root Directory: `tumor_classifier/frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. Add environment variables:
   - `VITE_API_URL`: Backend URL

### Model Deployment
1. Use Git LFS for model file:
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes
   git add model_weights.h5
   git commit -m "Add model file"
   git push origin main
   ```

## Testing

### Backend Testing
1. Health check:
   ```bash
   curl https://braindx-api.onrender.com/health
   ```

2. Model prediction:
   ```bash
   curl -X POST -F "file=@test_image.jpg" https://braindx-api.onrender.com/predict
   ```

### Frontend Testing
1. Local testing:
   - Run frontend locally
   - Test file upload
   - Verify API integration

2. Production testing:
   - Test deployed frontend
   - Verify CORS configuration
   - Check error handling

## Troubleshooting

### Common Issues
1. Model Loading Errors
   - Check model file path
   - Verify TensorFlow version
   - Check model file integrity

2. CORS Issues
   - Verify allowed origins in backend
   - Check frontend API URL configuration

3. Deployment Issues
   - Check build logs
   - Verify environment variables
   - Check dependency versions

### Logs
- Backend logs: Available in Render dashboard
- Frontend logs: Browser console
- Build logs: Vercel/Render deployment logs

## Maintenance
1. Regular updates:
   - Update dependencies
   - Monitor error logs
   - Check model performance

2. Backup:
   - Model file backup
   - Code repository backup
   - Environment variables backup

## Support
For issues or questions:
1. Check the troubleshooting guide
2. Review deployment logs
3. Contact development team 