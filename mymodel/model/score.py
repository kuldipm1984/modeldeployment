# score.py
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

import os
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model

def init():
    global model, sc, le_season, le_location, le_wt
    
    # Get model directory
    model_dir = Path(os.getenv('AZUREML_MODEL_DIR'))
    print(f"Model directory contents: {list(model_dir.glob('*'))}")  # Debug
    
    # Try multiple possible model paths
    possible_model_paths = [
        model_dir / 'classification_model.h5',  # Direct file
        model_dir / 'model' / 'classification_model.h5',  # Nested
        model_dir / '1' / 'classification_model.h5',  # Versioned
        next(model_dir.glob('**/*.h5')),  # Any .h5 file
        model_dir / 'model',  # SavedModel format
        model_dir / '1' / 'model'  # Versioned SavedModel
    ]
    
    # Load model
    model = None
    for model_path in possible_model_paths:
        try:
            if model_path.exists():
                print(f"Attempting to load model from: {model_path}")
                model = load_model(model_path)
                print(f"Successfully loaded model from: {model_path}")
                break
        except Exception as e:
            print(f"Failed to load from {model_path}: {str(e)}")
            continue
    
    if model is None:
        raise ValueError("Could not load model from any known location")
    
    # Load preprocessing objects
    preprocessors = {
        'sc.pkl': 'sc',
        'le_season.pkl': 'le_season',
        'le_location.pkl': 'le_location',
        'le_wt.pkl': 'le_wt'
    }
    
    for file, var_name in preprocessors.items():
        try:
            # Search recursively for the file
            file_path = next(model_dir.glob(f'**/{file}'), None)
            if file_path:
                with open(file_path, 'rb') as f:
                    globals()[var_name] = pickle.load(f)
                print(f"Loaded {file} from {file_path}")
            else:
                raise FileNotFoundError(f"{file} not found")
        except Exception as e:
            raise ValueError(f"Error loading {file}: {str(e)}")
    
    print("Initialization completed successfully")

def run(mini_batch):
    """Process each file in the mini_batch"""
    results = []
    
    for file_path in mini_batch:
        # Read the input data
        input_data = pd.read_csv(file_path)
        
        # Preprocess the data (same as your training preprocessing)
        # 1. Separate numerical and categorical features
        numerical_cols = ['Temperature', 'Humidity', 'Wind Speed', 
                         'Precipitation (%)', 'Atmospheric Pressure', 
                         'UV Index', 'Visibility (km)']
        x_numerical = input_data[numerical_cols]
        
        # Scale numerical features
        x_numerical_scaled = sc.transform(x_numerical)
        
        # 2. Process categorical columns
        input_data['Season'] = le_season.transform(input_data['Season'])
        input_data['Location'] = le_location.transform(input_data['Location'])
        
        # Extract categorical inputs
        x_season = input_data['Season'].values
        x_location = input_data['Location'].values
        
        # Make predictions
        predictions = model.predict([
            x_numerical_scaled,
            x_season,
            x_location
        ])
        
        # Convert predictions to original labels
        predicted_indices = np.argmax(predictions, axis=1)
        original_labels = le_wt.inverse_transform(predicted_indices)
        
        # Append results
        input_data['Prediction'] = original_labels
        results.append(input_data)
    
    # Return concatenated results
    return pd.concat(results)