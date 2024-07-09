import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load ridge regressor model and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Title for the Streamlit app
st.title("Predict 'Fire-Weather Index' [Shahid Anowar]")

# Input fields for user data
Temperature = st.number_input('Temperature in C', min_value=-22.0, max_value=42.0, step=0.1)
RH = st.number_input('Relative Humidity', min_value=21.0, max_value=90.0, step=0.1)
Ws = st.number_input('Wind Speed', min_value=6.0, max_value=29.0, step=0.1)
Rain = st.number_input('Rain', min_value=0.0, max_value=16.8, step=0.1)
FFMC = st.number_input('Fine Fuel Moisture Code', min_value=28.6, max_value=92.5, step=0.1)
DMC = st.number_input('Duff Moisture Code', min_value=1.1, max_value=65.9, step=0.1)
ISI = st.number_input('Drought Code', min_value=0.0, max_value=18.5, step=0.1)
Classes = st.number_input('Classes', min_value=0.0, max_value=1.0, step=1.0)
Region = st.number_input('Region', min_value=0.0, max_value=1.0, step=1.0)

# Button to make prediction
if st.button('Predict'):
    new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    result = ridge_model.predict(new_data_scaled)
    
    st.success(f'The prediction result is: {result[0]}')

