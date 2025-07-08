import streamlit as st
import joblib
import pandas as pd

# Load model and column info
model = joblib.load("pollution_model.pkl")  # This should return a model that gives multiple outputs
model_cols = joblib.load("model_columns.pkl")  # This should be a list like ['year', 'station_id']

pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Title and description
st.title(" Water Pollutants Predictor")
st.write("Predict water pollutants like O2, NO3, NO2, SO4, PO4, and CL based on Year and Station ID")

# Input fields
year_input = st.number_input("Enter Year", step=1, format="%d")
station_id = st.text_input("Enter Station ID", value="1")

# Predict button
if st.button("Predict"):
    if not station_id:
        st.warning("Please enter the Station ID.")
    else:
        # Prepare input DataFrame
        input_df = pd.DataFrame([[year_input, station_id]])

        # One-hot encode if needed
        input_encoded = pd.get_dummies(input_df)

        # Align with model columns # Add missing columns (if any) to match model_cols
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        # Reorder columns to match model
        input_encoded = input_encoded[model_cols]  # ensure correct column order

        # Predict pollutants
        predicted_pollutants = model.predict(input_encoded)[0]

        # Show results
        st.subheader(f"Predicted Pollutant Levels for Station '{station_id}' in {year_input}:")
        for pollutant, value in zip(pollutants, predicted_pollutants):
            st.write(f"{pollutant}: {value:.2f}")
