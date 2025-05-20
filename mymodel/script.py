from azureml.core import Run
new_run = Run.get_context()
ws = new_run.experiment.workspace

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2 as l2_regularizer  # Renamed to avoid conflict
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import pickle
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--l2_reg', type=float, default=0.01)
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--input_data', type=str)
args, _ = parser.parse_known_args()

# Get data
data = new_run.input_datasets['raw_data'].to_pandas_dataframe()
df=data.rename(columns={'Weather Type':'WeatherType'})

# Preprocessing
sc = StandardScaler()
df_numerical = df.select_dtypes(include=['int64', 'float64'])
df_numerical = sc.fit_transform(df_numerical)

le_season = LabelEncoder()
le_location = LabelEncoder()
le_wt = LabelEncoder()

df.Season = le_season.fit_transform(df.Season)
df.Location = le_location.fit_transform(df.Location)
df.WeatherType = le_wt.fit_transform(df.WeatherType)

x_num = df_numerical
x_cat1 = df.Season
x_cat2 = df.Location
y = df.WeatherType

x_num_train, x_num_test, x_cat1_train, x_cat1_test, x_cat2_train, x_cat2_test, y_train, y_test = train_test_split(
    x_num, x_cat1, x_cat2, y, test_size=0.2, random_state=42
)

def build_model(hp):
    num_seasons = df["Season"].nunique() + 1
    num_Locations = df["Location"].nunique() + 1
    embedding_dim = 10
    
    # Hyperparameters
    learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling='log')
    l2_reg = hp.Float('l2_reg', min_value=0.0001, max_value=0.01, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)    

    # Inputs 
    num_input = Input(shape=(x_num_train.shape[1],), name='num_input')
    cat1_input = Input(shape=(1,), name='cat1_input') 
    cat2_input = Input(shape=(1,), name='cat2_input')

    # Embedding 
    cat1_embedding = Embedding(num_seasons, embedding_dim, name="cat1_embedding")(cat1_input)
    cat2_embedding = Embedding(num_Locations, embedding_dim, name="cat2_embedding")(cat2_input)
    cat1_vec = Flatten()(cat1_embedding)
    cat2_vec = Flatten()(cat2_embedding)
    
    # Concatenate
    concat = Concatenate()([num_input, cat1_vec, cat2_vec])
   
    dense1 = Dense(256, activation='relu', kernel_regularizer=l2_regularizer(l2_reg))(concat)
    dense1 = Dropout(dropout_rate)(dense1)
    
    dense2 = Dense(128, activation='relu', kernel_regularizer=l2_regularizer(l2_reg))(dense1)
    dense2 = Dropout(dropout_rate)(dense2)

    dense3 = Dense(64, activation='relu', kernel_regularizer=l2_regularizer(l2_reg))(dense2)
    dense3 = Dropout(dropout_rate)(dense3)
    
    output = Dense(4, activation='softmax')(dense3)  
   
    model = Model(inputs=[num_input, cat1_input, cat2_input], outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'])
    
    return model

# Setup tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    directory='output/exp4',  # Changed to output directory
    project_name='weather_classification'
)

# Run tuning
tuner.search(
    [x_num_train, x_cat1_train, x_cat2_train],
    y_train,
    batch_size=50,
    epochs=10,
    validation_data=([x_num_test, x_cat1_test, x_cat2_test], y_test))
    
# Get best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate
test_loss, test_acc = best_model.evaluate([x_num_test, x_cat1_test, x_cat2_test], y_test)

# Log metrics
new_run.log('test_accuracy', test_acc)
new_run.log('test_loss', test_loss)

# Save artifacts
os.makedirs('outputs', exist_ok=True)  # Azure ML looks for 'outputs' folder by default

# Save preprocessing objects
with open('outputs/sc.pkl', 'wb') as f:
    pickle.dump(sc, f)
with open('outputs/le_season.pkl', 'wb') as f:
    pickle.dump(le_season, f)
with open('outputs/le_location.pkl', 'wb') as f:
    pickle.dump(le_location, f)
with open('outputs/le_wt.pkl', 'wb') as f:
    pickle.dump(le_wt, f)

# Save model
best_model.save('outputs/classification_model.h5')

# Complete the run
new_run.complete()