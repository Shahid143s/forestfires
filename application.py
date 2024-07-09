import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load ridge regressor model and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Title for the Streamlit app
st.title("Predict Datapoint using Ridge Regression")

# Input fields for user data
Temperature = st.number_input('Temperature', min_value=-50.0, max_value=50.0, step=0.1)
RH = st.number_input('RH', min_value=0.0, max_value=100.0, step=0.1)
Ws = st.number_input('Ws', min_value=0.0, max_value=100.0, step=0.1)
Rain = st.number_input('Rain', min_value=0.0, max_value=100.0, step=0.1)
FFMC = st.number_input('FFMC', min_value=0.0, max_value=100.0, step=0.1)
DMC = st.number_input('DMC', min_value=0.0, max_value=1000.0, step=0.1)
ISI = st.number_input('ISI', min_value=0.0, max_value=100.0, step=0.1)
Classes = st.number_input('Classes', min_value=0.0, max_value=10.0, step=0.1)
Region = st.number_input('Region', min_value=0.0, max_value=10.0, step=0.1)

# Button to make prediction
if st.button('Predict'):
    new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    result = ridge_model.predict(new_data_scaled)
    
    st.success(f'The prediction result is: {result[0]}')

