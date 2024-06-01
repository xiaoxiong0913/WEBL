import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model_path = "C:\\Users\\14701\\Desktop\\WEBL\\gbm_model.pkl"
scaler_path = "C:\\Users\\14701\\Desktop\\WEBL\\scaler.pkl"

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Define the feature names in the specified order
feature_names = [
    'Lym（10^9/L）',
    'Hb(g/L)',
    'Alb(g/L)',
    'reperfusiontherapy(yes1，no0)',
    'ECMO(yes1,no0)',
    'ACEI/ARB(yes1,no0)'
]

# Create the title for the web app
st.title('A machine learning-based model to predict 1-year risk among patients with acute myocardial infarction and cardiogenic shock ')

# Introduction section
st.markdown("""
## Introduction
This web-based calculator was developed based on the gradient boosting machine, with an AUC of 0.81 and Brier score of 0.1806. Users can obtain the one-year risk of death for a particular case by selecting parameters and clicking the "Calculate" button.
""")

# Create the input form
with st.form("prediction_form"):
    lym = st.slider('Lym (10^9/L)', min_value=0.0, max_value=8.0, value=1.0, step=0.1, key='Lym（10^9/L）')
    hb = st.slider('Hb (g/L)', min_value=0.0, max_value=200.0, value=100.0, step=1.0, key='Hb(g/L)')
    alb = st.slider('Alb (g/L)', min_value=0.0, max_value=50.0, value=25.0, step=0.1, key='Alb(g/L)')
    reperfusion_therapy = st.selectbox('Reperfusion Therapy', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='reperfusiontherapy(yes1，no0)')
    ecmo = st.selectbox('ECMO', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='ECMO(yes1,no0)')
    acei_arb = st.selectbox('ACEI/ARB', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='ACEI/ARB(yes1,no0)')

    submit_button = st.form_submit_button("Calculate")

# Process form submission
if submit_button:
    data = {
        "Lym（10^9/L）": lym,
        "Hb(g/L)": hb,
        "Alb(g/L)": alb,
        "reperfusiontherapy(yes1，no0)": reperfusion_therapy,
        "ECMO(yes1,no0)": ecmo,
        "ACEI/ARB(yes1,no0)": acei_arb,
    }

    try:
        # Convert input data to DataFrame using the exact feature names
        data_df = pd.DataFrame([data], columns=feature_names)

        # Scale the data using the loaded scaler
        data_scaled = scaler.transform(data_df)

        # Make a prediction
        prediction = model.predict_proba(data_scaled)[:, 1][0]  # Getting the probability of class 1
        st.write(f'Prediction: {prediction * 100:.2f}%')  # Convert probability to percentage
    except Exception as e:
        st.error(f'Error: {str(e)}')
