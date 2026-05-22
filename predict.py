import joblib
import pandas as pd
import numpy as np

def predict_mood(features_dict):
    """
    Predict the mood label for a new song given its features.
    
    Expected features_dict keys:
    'duration (ms)', 'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'spec_rate'
    """
    try:
        model = joblib.load('models/logistic_regression_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        print("Model or scaler not found. Please run train_model.py first.")
        return None
        
    # Define the exact order of features that the model expects
    feature_order = [
        'duration (ms)', 'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'spec_rate'
    ]
    
    # Create a DataFrame for the input
    input_df = pd.DataFrame([features_dict], columns=feature_order)
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Map the integer label to a human-readable mood based on data analysis
    mood_map = {
        0: "Calm / Relaxed",
        1: "Happy / Upbeat",
        2: "Energetic / Intense",
        3: "Sad / Melancholic"
    }
    
    predicted_mood = mood_map.get(prediction, "Unknown")
    
    print(f"Predicted Mood Label: {prediction} ({predicted_mood})")
    print("Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  Class {model.classes_[i]} ({mood_map.get(model.classes_[i])}): {prob:.4f}")
        
    return predicted_mood

if __name__ == "__main__":
    # Example usage with some dummy values
    sample_song = {
        'duration (ms)': 210000.0,
        'danceability': 0.75,
        'energy': 0.85,
        'loudness': -5.0,
        'speechiness': 0.05,
        'acousticness': 0.1,
        'instrumentalness': 0.0,
        'liveness': 0.15,
        'valence': 0.8,
        'tempo': 120.0,
        'spec_rate': 0.0000005
    }
    
    print("Testing inference with a sample song...")
    predict_mood(sample_song)
