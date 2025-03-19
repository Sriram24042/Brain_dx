# BrainDx - Brain Tumor Classification

A web application for brain tumor classification using deep learning.

## Features

- Upload brain MRI images
- Real-time tumor classification
- Modern, responsive UI
- Secure file handling
- High-accuracy predictions

## Tech Stack

### Frontend
- React with TypeScript
- Material-UI
- Framer Motion
- Axios for API calls

### Backend
- FastAPI
- TensorFlow
- Python 3.8+
- Scikit-learn

## Project Structure

```
Brain_dx/
├── tumor_classifier/
│   ├── backend/      # FastAPI server
│   └── frontend/     # React application
```

## Local Development

### Backend Setup
```bash
cd tumor_classifier/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup
```bash
cd tumor_classifier/frontend
npm install
npm run dev
```

## Deployment

- Backend: Deployed on Render
- Frontend: Deployed on Vercel

## API Endpoints

- `POST /predict`: Upload and classify brain MRI images
- `GET /health`: Health check endpoint

## License

MIT License 