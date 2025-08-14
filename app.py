import streamlit as st
import numpy as np
import joblib


scaler = joblib.load("scaler.pkl")
model = joblib.load("ann_model.pkl")


features = [
    ("radius_mean", 5.0, 35.0),
    ("texture_mean", 5.0, 45.0),
    ("perimeter_mean", 30.0, 200.0),
    ("area_mean", 50.0, 3000.0),
    ("smoothness_mean", 0.03, 0.2),
    ("compactness_mean", 0.0, 0.5),
    ("concavity_mean", 0.0, 0.5),
    ("concave points_mean", 0.0, 0.25),
    ("symmetry_mean", 0.09, 0.35),
    ("fractal_dimension_mean", 0.04, 0.12),

    ("radius_se", 0.05, 3.5),
    ("texture_se", 0.1, 6.0),
    ("perimeter_se", 0.5, 25.0),
    ("area_se", 5.0, 600.0),
    ("smoothness_se", 0.0005, 0.04),
    ("compactness_se", 0.0, 0.15),
    ("concavity_se", 0.0, 0.45),
    ("concave points_se", 0.0, 0.06),
    ("symmetry_se", 0.005, 0.09),
    ("fractal_dimension_se", 0.0005, 0.04),

    ("radius_worst", 5.0, 40.0),
    ("texture_worst", 5.0, 55.0),
    ("perimeter_worst", 30.0, 270.0),
    ("area_worst", 50.0, 5000.0),
    ("smoothness_worst", 0.05, 0.25),
    ("compactness_worst", 0.02, 1.2),
    ("concavity_worst", 0.0, 1.3),
    ("concave points_worst", 0.0, 0.35),
    ("symmetry_worst", 0.15, 0.7),
    ("fractal_dimension_worst", 0.05, 0.25),
]


st.set_page_config(page_title="BreastCancerPredictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>🩺 Breast Cancer Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "### Provide Your Tumor Measurement Details\n"
    "Enter your **medical measurements** in the fields below. "
    "The model will predict whether the tumor is **Benign (non-cancerous)** or **Malignant (cancerous)**."
)

cols = st.columns(3)
inputs = []

for i, (feature, min_val, max_val) in enumerate(features):
    col = cols[i // 10]  # Group by 10 per column
    with col:
        val = st.number_input(
            f"{feature.replace('_', ' ').title()}",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2),
            step=(max_val - min_val) / 200
        )
        inputs.append(val)


if st.button("🔮 Predict"):

    X_input = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    
    pred_prob = model.predict(X_scaled)[0][0]
    pred_class = int(round(pred_prob))  # 0 = Benign, 1 = Malignant

    
    if pred_class == 0:
        st.success("✅ **Prediction: Benign (Non-cancerous)**")
    else:
        st.error("⚠️ **Prediction: Malignant (Cancerous)**")
