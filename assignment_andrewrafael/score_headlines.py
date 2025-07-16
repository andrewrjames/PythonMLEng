#!/usr/bin/env python3
"""
Sentiment Analysis Script for News Headlines

This script processes news headlines from a text file and generates sentiment scores
using a pre-trained SVM model. It outputs results in the format:
sentiment_label, original_headline

Usage:
    python score_headlines.py <input_file> <source>

Arguments:
    input_file: Path to text file containing headlines (one per line)
    source: Source identifier (e.g., 'nyt', 'chicagotribune')

Example:
    python score_headlines.py headlines_nyt_2024-12-02.txt nyt
"""

import sys
import os
import argparse
from datetime import datetime
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


def load_headlines(file_path):
    """Load headlines from a text file in the data/raw folder, one headline per line."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the project root (one level up from assignment1_andrewrafael/)
    project_root = os.path.dirname(script_dir)
    # Construct path to data/raw folder
    raw_data_path = os.path.join(project_root, 'data', 'raw', file_path)
    
    try:
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            headlines = [line.strip() for line in f if line.strip()]
        return headlines
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found in data/raw folder.")
        print(f"Looking for: {raw_data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)

def load_model(model_path='svm.joblib'):
    """Load the pre-trained SVM model from the models folder."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the project root (one level up from assignment1_andrewrafael/)
    project_root = os.path.dirname(script_dir)
    # Construct path to models folder
    models_path = os.path.join(project_root, 'models', model_path)
    
    try:
        model = joblib.load(models_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found in models folder.")
        print(f"Looking for: {models_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def vectorize_headlines(headlines):
    """Convert headlines to embeddings using SentenceTransformer."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(headlines)
        return embeddings
    except Exception as e:
        print(f"Error vectorizing headlines: {e}")
        sys.exit(1)


def predict_sentiment(embeddings, svm_model):
    """Predict sentiment labels for the given embeddings."""
    try:
        predictions = svm_model.predict(embeddings)
        return predictions
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        sys.exit(1)


def generate_output_filename(source):
    """Generate output filename based on source and current date."""
    today = datetime.now()
    year = today.year
    month = today.month
    day = today.day
    return f"headline_scores_{source}_{year}_{month:02d}_{day:02d}.txt"


def save_results(headlines, predictions, output_file):
    """Save results to output file in the data/processed folder."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the project root (one level up from assignment1_andrewrafael/)
    project_root = os.path.dirname(script_dir)
    # Construct path to data/processed folder
    processed_data_path = os.path.join(project_root, 'data', 'processed', output_file)
    
    try:
        with open(processed_data_path, 'w', encoding='utf-8') as f:
            for headline, prediction in zip(headlines, predictions):
                f.write(f"{prediction}, {headline}\n")
        print(f"Results saved to: {processed_data_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)


def main():
    """Main function to orchestrate the sentiment analysis process."""
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of news headlines using pre-trained SVM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python score_headlines.py headlines_nyt_2024-12-02.txt nyt
    python score_headlines.py headlines_chicagotribune_2024-12-01.txt chicagotribune
        """
    )
    
    parser.add_argument('input_file', help='Path to input text file containing headlines')
    parser.add_argument('source', help='Source identifier (e.g., nyt, chicagotribune)')
    
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nError: Please provide both input file and source parameters.")
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Note: File existence will be checked in load_headlines function
    # since it constructs the full path to data/raw folder
    
    print(f"Processing headlines from: {args.input_file}")
    print(f"Source: {args.source}")
    
    # Load headlines
    headlines = load_headlines(args.input_file)
    print(f"Loaded {len(headlines)} headlines")
    print("▶ load_headlines returned:", headlines[:5], "…")
    
    # Load the SVM model
    svm_model = load_model()
    print("SVM model loaded successfully")
    
    # Vectorize headlines
    print("Converting headlines to embeddings...")
    embeddings = vectorize_headlines(headlines)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Predict sentiment
    print("Predicting sentiment...")
    predictions = predict_sentiment(embeddings, svm_model)
    print("▶ sample predictions:", predictions[:5], "…")
    
    # Generate output filename
    output_file = generate_output_filename(args.source)
    
    # Save results
    save_results(headlines, predictions, output_file)
    
    # Print summary statistics
    unique_predictions, counts = np.unique(predictions, return_counts=True)
    print("\nSentiment Analysis Summary:")
    for sentiment, count in zip(unique_predictions, counts):
        print(f"  {sentiment}: {count} headlines")
    
    print(f"\nTotal processed: {len(headlines)} headlines")


if __name__ == "__main__":
    main()