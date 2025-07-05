import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('xgb_wine_quality_model_retrained.pkl')

# App title
st.title("Wine Quality Identifier 🍷")
st.markdown("Enter the chemical attributes of a red wine sample to identify its quality.")

# Input form
with st.form("input_form"):
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, format="%.2f")
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, format="%.2f")
    citric_acid = st.number_input("Citric Acid", min_value=0.0, format="%.2f")
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, format="%.2f")
    chlorides = st.number_input("Chlorides", min_value=0.0, format="%.4f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, format="%.1f")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, format="%.1f")
    density = st.number_input("Density", min_value=0.0, format="%.5f")
    pH = st.number_input("pH", min_value=0.0, format="%.2f")
    sulphates = st.number_input("Sulphates", min_value=0.0, format="%.2f")
    alcohol = st.number_input("Alcohol", min_value=0.0, format="%.2f")
    
    submitted = st.form_submit_button("Identify Quality")

# Predict
if submitted:
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][1] if prediction == 1 else model.predict_proba(input_data)[0][0]

    st.subheader("Result")
    if prediction == 1:
        st.success(f"✅ **Good Quality Wine**\n\nConfidence: {confidence:.2%}")
    else:
        st.error(f"❌ **Not Good Quality Wine**\n\nConfidence: {confidence:.2%}")
