import pandas as pd
import pickle as pk
import streamlit as st

# Load the model
with open('rf_model.pkl', 'rb') as file:
    rf_model = pk.load(file)

# Load data
data = pd.read_csv("Cardetails (1).csv")

# Extract brand names
def brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
data['name'] = data['name'].apply(brand_name)

# Set the Streamlit layout
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        font-family: 'Arial', sans-serif;
    }
    .container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #007BFF;
        font-size: 36px;
        text-align: center;
        margin-bottom: 20px;
    }
    h2 {
        color: #28A745;
        text-align: center;
    }
    label {
        font-weight: bold;
        color: #333333;
    }
    button {
        background-color: #007BFF !important;
        color: #ffffff !important;
        font-size: 18px !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Container for the app
st.markdown("<div class='container'>", unsafe_allow_html=True)

# Header
st.markdown("<h1>🚗 Car Price Prediction</h1>", unsafe_allow_html=True)

# Input Section
st.subheader("Enter Car Details:")

col1, col2, col3 = st.columns(3)

with col1:
    name = st.selectbox('Select Car Brand', data['name'].unique())
    fuel = st.selectbox('Fuel Type', data['fuel'].unique())
    transmission = st.selectbox('Transmission Type', data['transmission'].unique())
    mileage = st.slider('Car Mileage (km/l)', 10, 40)

with col2:
    year = st.slider('Car Manufactured Year', 1995, 2024)
    seller_type = st.selectbox('Seller Type', data['seller_type'].unique())
    owner = st.selectbox('Owner', data['owner'].unique())
    engine = st.number_input("Engine CC", 700, 5000, step = 50)

with col3:
    km_driven = st.number_input('No of Kms Driven', min_value=20, max_value=1000000, step=100)
    max_power = st.slider('Max Power (BHP)', 0, 200)
    seats = st.slider('No. of Seats', 4, 10)

# Predict Button
if st.button("🔍 Predict Price"):
    # Prepare input dataframe
    input = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    st.write(input)
    # Encode categorical features
    input["owner"] = input['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                             'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5])

    input["fuel"] = input['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4])
    input["seller_type"] = input['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3])
    input['transmission'] = input['transmission'].replace(['Manual', 'Automatic'], [1, 2]).astype(int)

    input["name"] = input['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                           'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                           'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                           'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                           'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                          list(range(1, 32)))

    # Display input for transparency
    st.write("🚘 **Your Input:**")
    

    # Price Prediction
    with st.spinner("Calculating car price..."):
        car_price = rf_model.predict(input)

    st.markdown(f"<h2>💰 Estimated Price: ₹{car_price[0]:,.2f}</h2>", unsafe_allow_html=True)

# Close container div
st.markdown("</div>", unsafe_allow_html=True)
