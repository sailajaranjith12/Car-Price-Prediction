import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/sailajaranjith12/Car-Price-Prediction/refs/heads/main/Dataset/cleaned_car_data.csv"
df = pd.read_csv(url)

# Load trained model
model = joblib.load("models/linear_regression_model.pkl")

st.title("🚗 Car Price Prediction App")
st.markdown("### Predict the selling price of a used car based on its features.")

# User inputs
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2024, step=1, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000, value=50000)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, step=0.5, value=20.0)
engine = st.number_input("Engine Capacity (CC)", min_value=600, max_value=5000, step=50, value=1200)
max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, step=5.0, value=90.0)
seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8])

# Encoding categorical inputs
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "LPG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# Convert categorical inputs to match model encoding
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_lpg = 1 if fuel_type == "LPG" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0
seller_individual = 1 if seller_type == "Individual" else 0
seller_trustmark = 1 if seller_type == "Trustmark Dealer" else 0
trans_manual = 1 if transmission == "Manual" else 0
owner_second = 1 if owner == "Second Owner" else 0
owner_third = 1 if owner == "Third Owner" else 0
owner_fourth = 1 if owner == "Fourth & Above Owner" else 0
owner_test_drive = 1 if owner == "Test Drive Car" else 0

# Create a feature array
input_data = np.array([[year, km_driven, mileage, engine, max_power, seats,
                        fuel_diesel, fuel_lpg, fuel_petrol,
                        seller_individual, seller_trustmark,
                        trans_manual, owner_fourth, owner_second, owner_test_drive, owner_third]])

# Ensure feature count matches trained model
if len(input_data[0]) != len(df.drop(columns=['selling_price']).columns):
    st.error(f"Feature mismatch! Model expects {len(df.drop(columns=['selling_price']).columns)} features, but received {len(input_data[0])}.")
    st.stop()

# Predict price
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Selling Price: ₹{predicted_price:,.2f}")
