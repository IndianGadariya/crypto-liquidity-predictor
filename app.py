import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and encoder
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")

st.title("🔮 Cryptocurrency Liquidity Predictor")
st.markdown("Predict whether a cryptocurrency has **Low**, **Medium**, or **High** liquidity level.")

# Input form
with st.form("input_form"):
    price = st.number_input("Current Price in USD", min_value=0.0)
    h1 = st.number_input("% Price Change in Last 1 Hour (e.g. 0.01 for +1%)")
    h24 = st.number_input("% Price Change in Last 24 Hours")
    d7 = st.number_input("% Price Change in Last 7 Days")
    volume_24h = st.number_input("24-Hour Trading Volume (USD)", min_value=0.0)
    market_cap = st.number_input("Market Capitalization (USD)", min_value=0.0)
    volatility = st.number_input("Volatility (standard deviation of price changes)")
    trend_up = st.selectbox("Is the 24h Trend Positive?", [0, 1])

    submit = st.form_submit_button("Predict")

if submit:
    if market_cap == 0 or price == 0:
        st.error("Market cap and price must be greater than 0 to calculate derived features.")
    else:
        # Auto-calculated features
        liquidity_ratio = volume_24h / market_cap
        trend_strength = h1 + h24 + d7
        log_price = np.log1p(price)

        features = np.array([[price, h1, h24, d7, volume_24h, market_cap,
                              liquidity_ratio, volatility, trend_strength,
                              trend_up, log_price]])

        # Scale
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)
        label = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🧠 Predicted Liquidity Level: **{label}**")
