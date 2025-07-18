#serve post json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List 
import os
import joblib
from sentence_transformers import SentenceTransformer
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models once
def load_svm_model(model_path = 'svm.joblib'):
    """Load the pre-trained SVM model from the models folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_path = os.path.join(project_root, 'models', model_path)
    
    try:
        model = joblib.load(models_path)
        return model
    except FileNotFoundError:
        logger.error(f"Error: Model file '{model_path}' not found in models folder: {models_path}.")
        
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

svm_model = load_svm_model()
logger.info("SVM model loaded successfully.")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model loaded successfully.")

@app.get('/status')
def status():
    d = {'status': 'OK'}
    return d

class HeadlineData(BaseModel):
    headlines: List[str]

@app.post('/score_headlines')
def score_headlines(client_props: HeadlineData):
    try:
        logger.info(f"Processing {len(client_props.headlines)} headlines for scoring.")

        # Vectorize headlines using sentence transformer models
        embeddings = embedding_model.encode(client_props.headlines)

        # Predict using SVM Model
        predictions = svm_model.predict(embeddings)

        # Convert predictions to list
        predictions_labels = predictions.tolist()

        return {'labels': predictions_labels}
    except Exception as e:
        logger.error(f"Error scoring headlines: {e}")
        return {'error': str(e)}

# fastapi dev score_headlines_api.py --host 0.0.0.0 --port 8009 