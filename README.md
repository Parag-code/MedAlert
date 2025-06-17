# 🩺 MedAlert – Emergency Triage Assistant

**MedAlert** is an AI-powered emergency triage assistant built with Streamlit. It predicts the urgency level of a patient (Non-Urgent, Urgent, Critical) based on key medical indicators such as age, gender, pain presence, and mental status. It is designed to assist healthcare professionals in making quick, data-driven decisions during emergencies.

---

## 🧠 Features

- 🎯 **Urgency Level Prediction**: Classifies patients into `Non-Urgent`, `Urgent`, or `Critical` using a trained ML model.
- 🧍 **User Inputs**:
  - **Age** (1–100)
  - **Gender**: Male / Female
  - **Pain**: Yes / No
  - **Mental Status**: Alert / Confused
- 📊 **Real-time Results**: Instant output with clear, color-coded feedback:
  - 🟢 Non-Urgent
  - 🟠 Urgent
  - 🔴 Critical
- 📈 **Preprocessing & Scaling**: All features are scaled and encoded before prediction.
- 🖼️ **Custom Logo & Theming**: Clean UI with a center-aligned MedAlert logo.

---

## 🏗️ Tech Stack

- **Frontend**: [Streamlit]
- **Backend**: Python, scikit-learn
- **Model Deployment**: joblib for loading ML models
- **ML Model**: Classification model trained to predict triage urgency level

---
