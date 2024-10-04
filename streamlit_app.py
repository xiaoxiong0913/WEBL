import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Define the feature names in the specified order
feature_names = [
    'Lym（10^9/L）',
    'Hb(g/L)',
    'Alb(g/L)',
    'reperfusiontherapy(yes1，no0)',
    'ECMO(yes1,no0)',
    'ACEI/ARB(yes1,no0)'
]

# Load the model and scaler
model_path = "gbm_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Define risk cutoff and thresholds
risk_cutoff = 0.479

# Create the title for the web app
st.title(
    'A machine learning-based model to predict 1-year risk among patients with acute myocardial infarction and cardiogenic shock ')

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
    reperfusion_therapy = st.selectbox('Reperfusion Therapy', options=[0, 1],
                                       format_func=lambda x: 'Yes' if x == 1 else 'No',
                                       key='reperfusiontherapy(yes1，no0)')
    ecmo = st.selectbox('ECMO', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='ECMO(yes1,no0)')
    acei_arb = st.selectbox('ACEI/ARB', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No',
                            key='ACEI/ARB(yes1,no0)')
    beta_blocker = st.selectbox('β-receptor Blocker', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    surgery = st.selectbox('Surgery Therapy', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    submit_button = st.form_submit_button("Predict")

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
        # Convert input data to DataFrame using the exact feature names and order
        data_df = pd.DataFrame([data], columns=feature_names)

        # Scale the data using the loaded scaler
        data_scaled = scaler.transform(data_df)

        # Make a prediction
        prediction = model.predict_proba(data_scaled)[:, 1][0]  # Getting the probability of class 1

        # Display prediction result
        st.subheader("Prediction Result:")
        st.write(f'Prediction: {prediction * 100:.2f}%')

        # Risk stratification and personalized recommendations
        if prediction >= risk_cutoff:
            st.markdown("<span style='color:red'>High risk: This patient is classified as a high-risk patient.</span>",
                        unsafe_allow_html=True)
            st.subheader("Personalized Recommendations:")

            # Recommendations for numerical values
            if lym < 0.8:
                st.markdown(
                    f"<span style='color:red'>Lym (10^9/L): Your value is {lym}. It is lower than the normal range (0.8 - 4.0). Consider increasing it towards 0.8.</span>",
                    unsafe_allow_html=True)
            elif lym > 4.0:
                st.markdown(
                    f"<span style='color:red'>Lym (10^9/L): Your value is {lym}. It is higher than the normal range (0.8 - 4.0). Consider decreasing it towards 4.0.</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"Lym (10^9/L): Your value is within the normal range (0.8 - 4.0).")

            if hb < 120:
                st.markdown(
                    f"<span style='color:red'>Hb (g/L): Your value is {hb}. It is lower than the normal range (120 - 170). Consider increasing it towards 120.</span>",
                    unsafe_allow_html=True)
            elif hb > 170:
                st.markdown(
                    f"<span style='color:red'>Hb (g/L): Your value is {hb}. It is higher than the normal range (120 - 170). Consider decreasing it towards 170.</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"Hb (g/L): Your value is within the normal range (120 - 170).")

            if alb < 35:
                st.markdown(
                    f"<span style='color:red'>Alb (g/L): Your value is {alb}. It is lower than the normal range (35 - 50). Consider increasing it towards 35.</span>",
                    unsafe_allow_html=True)
            elif alb > 50:
                st.markdown(
                    f"<span style='color:red'>Alb (g/L): Your value is {alb}. It is higher than the normal range (35 - 50). Consider decreasing it towards 50.</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"Alb (g/L): Your value is within the normal range (35 - 50).")

            # Recommendations for treatments and therapies
            if beta_blocker == 0:
                st.write("Consider using β-receptor blocker medication.")
            if acei_arb == 0:
                st.write("Consider using ACEI/ARB medication.")
            if surgery == 0:
                st.write("Consider undergoing surgery therapy.")
            if reperfusion_therapy == 0:
                st.write("Consider undergoing reperfusion therapy.")
            if ecmo == 0:
                st.write("Consider ECMO therapy for better management.")

        else:
            st.markdown("<span style='color:green'>Low risk: This patient is classified as a low-risk patient.</span>",
                        unsafe_allow_html=True)

    except Exception as e:
        st.error(f'Error: {str(e)}')
