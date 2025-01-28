import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the dataset (for unique values in dropdowns)
car = pd.read_csv('Cleaned_Car.csv')

# Load the trained model
with open('LinearRegressionModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Title and description of the app
st.title("Car Price Prediction App")
st.write("Enter the details of the car to get the predicted price.")

# Input fields for car features
car_model = st.selectbox("Car Model", options=car['name'].unique())
fuel_type = st.selectbox("Fuel Type", options=car['fuel_type'].unique())
company = st.selectbox("Company", options=car['company'].unique())
km_driven = st.number_input("Kilometers Driven", min_value=0, step=100, format="%d")
year = st.number_input("Year of Manufacture", min_value=1995, max_value=2025, step=1)

# Predict button
if st.button("Predict Price"):
    
    
    predicted_price=model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],
                              data=np.array([car_model, company, year, km_driven, fuel_type]).reshape(1,5)))
   

    print(predicted_price[0])
    
    # Display the predicted price
    st.success(f"The predicted price of the car is: ₹{predicted_price[0]:,.2f}")
