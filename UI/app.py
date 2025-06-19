import streamlit as st
import numpy as np
import joblib
import os

# Set page configuration first
st.set_page_config(page_title="MedAlert", page_icon="🩺")

# Load saved models
MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), "models"))

model = joblib.load(os.path.join(MODELS_DIR, "medalert_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "medalert_scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "medalert_label_encoder.pkl"))

# App UI

# Display logo
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("UI/logo.png", width=200)

st.title("MedAlert – Emergency Triage Assistant")
st.markdown("Predict patient urgency level based on their symptoms.")

st.sidebar.header("Enter Patient Details")

# Input fields
age = st.sidebar.slider("Age", 1, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
pain = st.sidebar.radio("Is the patient in pain?", ["No", "Yes"])

with st.sidebar.expander("ℹ️ What does 'Mental Status' mean?"):
    st.markdown("""
    - **Alert**: Fully aware and responsive
    - **Confused**: Disoriented or slow to respond
    """)
mental_status = st.sidebar.selectbox("Mental Status", ["Alert", "Confused"])

# Encode input
gender_val = 0 if gender == "Female" else 1
pain_val = 0 if pain == "No" else 1
mental_val = 0 if mental_status == "Alert" else 1

features = np.array([[age, gender_val, pain_val, mental_val]])
features_scaled = scaler.transform(features)

# Prediction
if st.button("Predict Urgency Level"):
    prediction = model.predict(features_scaled)
    urgency = label_encoder.inverse_transform(prediction)[0]

    st.subheader("🩹 Prediction Result:")
    if urgency == "Non-Urgent":
        st.success("🟢 Non-Urgent: No immediate action needed.")
    elif urgency == "Urgent":
        st.warning("🟠 Urgent: Needs medical attention soon.")
    else:
        st.error("🔴 Critical: Immediate intervention required!")
