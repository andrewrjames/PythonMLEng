#serve post json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List 
import os
import joblib
from sentence_transformers import SentenceTransformer


app = FastAPI()

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
        print(f"Error: Model file '{model_path}' not found in models folder.")
        print(f"Looking for: {models_path}")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

svm_model = load_svm_model()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get('/status')
def status():
    d = {'status': 'OK'}
    return d

class HeadlineData(BaseModel):
    headlines: List[str]

@app.post('/score_headlines')
def score_headlines(client_props: HeadlineData):
    try:
        # Vectorize headlines using sentence transformer models
        embeddings = embedding_model.encode(client_props.headlines)

        # Predict using SVM Model
        predictions = svm_model.predict(embeddings)

        # Convert predictions to list
        predictions_labels = predictions.tolist()

        return {'labels': predictions_labels}
    except Exception as e:
        return {'error': str(e)}

# fastapi dev serve_post_json.py