# score.py
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

import os
from tensorflow.keras.models import load_model

def init():
    global model, sc, le_season, le_location, le_wt
    
    model_dir = os.getenv('AZUREML_MODEL_DIR')  
   
    model_path = os.path.join(model_dir, 'classification_model.h5')
    model = load_model(model_path)
    
    sc_path = os.path.join(model_dir, 'sc.pkl')
    season_path = os.path.join(model_dir, 'le_season.pkl') 
    location_path = os.path.join(model_dir, 'le_location.pkl')
    wt_path = os.path.join(model_dir, 'le_wt.pkl')
    
    with open(sc_path, 'rb') as f:
        sc = pickle.load(f)
    with open(season_path, 'rb') as f:
        le_season = pickle.load(f)
    with open(location_path, 'rb') as f:
        le_location = pickle.load(f)
    with open(wt_path, 'rb') as f:
        le_wt = pickle.load(f)

def run(mini_batch):
    
    results = []
    
    for file_path in mini_batch:
        
        input_data = pd.read_csv(file_path)        
     
        numerical_cols = ['Temperature', 'Humidity', 'Wind Speed', 
                         'Precipitation (%)', 'Atmospheric Pressure', 
                         'UV Index', 'Visibility (km)']
        x_numerical = input_data[numerical_cols]        
       
        x_numerical_scaled = sc.transform(x_numerical)        
        
        input_data['Season'] = le_season.transform(input_data['Season'])
        input_data['Location'] = le_location.transform(input_data['Location'])        
        
        x_season = input_data['Season'].values
        x_location = input_data['Location'].values        
        
        predictions = model.predict([
            x_numerical_scaled,
            x_season,
            x_location
        ])        
        
        predicted_indices = np.argmax(predictions, axis=1)
        original_labels = le_wt.inverse_transform(predicted_indices)        
        
        input_data['Prediction'] = original_labels
        results.append(input_data)    
    
    return pd.concat(results)