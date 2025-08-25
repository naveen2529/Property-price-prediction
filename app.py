import streamlit as st
import pandas as pd
import joblib
import re
from difflib import get_close_matches

# --------------------------
# Load model & preprocessors
# --------------------------
model = joblib.load("gradient_boosting_tuned.pkl")
scaler = joblib.load("robust_scaler.pkl")
transformers = joblib.load("yeo_transformers.pkl")   # dict of Yeo-Johnson transformers
encoders = joblib.load("label_encoders.pkl")         # dict of LabelEncoders

# Columns to apply Yeo-Johnson + Scaler
numeric_cols = ["UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT",
                "READY_TO_MOVE", "RESALE", "LONGITUDE", "LATITUDE"]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")
st.title("🏡 House Price Prediction (in Lakhs)")

st.markdown("### 📌 Enter property details below")

# Address input
address = st.text_input("📍 Full Address (City will be auto-detected):")

# Posted By - radio buttons
posted_by = st.radio("👤 Posted By", encoders["POSTED_BY"].classes_)

# Binary choices as radios
under_construction = st.radio("🏗️ Under Construction", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
rera = st.radio("📜 RERA Approved", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
ready_to_move = st.radio("🚚 Ready to Move", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
resale = st.radio("🔁 Resale", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

# BHK number with slider
bhk_no = st.slider("🛏️ Number of Bedrooms (BHK)", min_value=1, max_value=20, step=1, value=2)

# BHK or RK as radio
bhk_or_rk = st.radio("🏠 Type", encoders["BHK_OR_RK"].classes_)

# Square Feet with slider
square_ft = st.slider("📐 Square Feet", min_value=100, max_value=10000, step=10, value=1000)

# Longitude and Latitude sliders
longitude = st.slider("🌍 Longitude", min_value=-180.0, max_value=180.0, value=77.0, step=0.001)
latitude = st.slider("🌍 Latitude", min_value=-90.0, max_value=90.0, value=28.0, step=0.001)

# --------------------------
# Detect city from address
# --------------------------
city = None
if address:
    addr_clean = address.strip().lower()

    # Exact/substring match
    for c in encoders["CITY"].classes_:
        c_clean = c.strip().lower()
        if re.search(rf"\b{c_clean}\b", addr_clean):
            city = c
            break

    # Fuzzy match if exact not found
    if not city:
        matches = get_close_matches(addr_clean, [c.lower() for c in encoders["CITY"].classes_], n=1, cutoff=0.7)
        if matches:
            city = [c for c in encoders["CITY"].classes_ if c.lower() == matches[0]][0]

# --------------------------
# Build input dataframe
# --------------------------
if not city:
    st.warning("⚠️ Could not detect city from address. Please include a valid city name.")
else:
    st.success(f"📍 City detected: **{city}**")

    raw_input = pd.DataFrame({
        "POSTED_BY": [posted_by],
        "UNDER_CONSTRUCTION": [under_construction],
        "RERA": [rera],
        "BHK_NO.": [bhk_no],
        "BHK_OR_RK": [bhk_or_rk],
        "SQUARE_FT": [square_ft],
        "READY_TO_MOVE": [ready_to_move],
        "RESALE": [resale],
        "LONGITUDE": [longitude],
        "LATITUDE": [latitude],
        "CITY": [city]
    })

    # --------------------------
    # Preprocessing
    # --------------------------
    # Label encode categorical
    for col, le in encoders.items():
        raw_input[col] = le.transform(raw_input[col])

    # Yeo-Johnson transform
    for col, pt in transformers.items():
        raw_input[[col]] = pt.transform(raw_input[[col]])

    # RobustScaler
    raw_input[numeric_cols] = scaler.transform(raw_input[numeric_cols])

    # --------------------------
    # Prediction
    # --------------------------
    if st.button("🔮 Predict Price"):
        prediction = model.predict(raw_input)[0]
        st.success(f"🏠 Predicted Price: **{prediction:.2f} Lakhs**")
