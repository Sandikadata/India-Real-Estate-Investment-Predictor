import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np

st.set_page_config(page_title="Real Estate Investment Predictor",
                   page_icon="🏠", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { border-radius: 8px; height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)


st.title("🏠 Real Estate Investment Predictor")
st.markdown("Enter property details to evaluate investment potential and future valuation.")


# ── Load Models and Separate Column Lists ────────────────
@st.cache_resource
def load_assets():
    # This finds the absolute path of the folder where main.py is located
    base_path = os.path.dirname(__file__)

    # Create absolute paths for each file
    clf_path = os.path.join(base_path, "classifier.pkl")
    reg_path = os.path.join(base_path, "regressor.pkl")
    clf_cols_path = os.path.join(base_path, "clf_columns.pkl")
    reg_cols_path = os.path.join(base_path, "reg_columns.pkl")

    # Load them using the full path
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    clf_columns = joblib.load(clf_cols_path)
    reg_columns = joblib.load(reg_cols_path)

    return clf, reg, clf_columns, reg_columns


try:
    clf, reg, clf_columns, reg_columns = load_assets()
except Exception as e:
    st.error(f"Technical Debug Info: {e}")
    st.write(f"Current Directory: {os.getcwd()}")
    st.stop()

# ── Input UI ────────────────────────────────────────────
st.subheader("📋 Property Details")
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Current Price (Lakhs)", 10, 1000, 50)
    size = st.slider("Size (SqFt)", 300, 10000, 1200)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5, 6])
    age_cat = st.selectbox("Property Age",
                           ["0-5 Years", "5-10 Years",
                            "10-20 Years", "20+ Years"])
    amenities = st.slider("Amenities Count", 1, 15, 5)

with col2:
    transport = st.slider("Transport Score (0:Low - 2:High)", 0, 2, 1)
    schools = st.number_input("Nearby Schools", 0, 50, 5)
    hospitals = st.number_input("Nearby Hospitals", 0, 50, 3)
    city = st.selectbox("City", sorted([
        "Ahmedabad", "Amritsar", "Bangalore", "Bhopal",
        "Bhubaneswar", "Bilaspur", "Chennai", "Coimbatore",
        "Cuttack", "Dehradun", "Durgapur", "Dwarka",
        "Faridabad", "Gaya", "Gurgaon", "Guwahati",
        "Haridwar", "Hyderabad", "Indore", "Jaipur",
        "Jamshedpur", "Jodhpur", "Kochi", "Kolkata",
        "Lucknow", "Ludhiana", "Mangalore", "Mumbai",
        "Mysore", "Nagpur", "New Delhi", "Noida",
        "Patna", "Pune", "Raipur", "Ranchi", "Silchar",
        "Surat", "Trivandrum", "Vijayawada",
        "Vishakhapatnam", "Warangal"]))
    prop_type = st.selectbox("Property Type",
                             ["Apartment", "Independent House", "Villa"])
    furnished = st.selectbox("Furnished Status",
                             ["Furnished", "Semi-furnished", "Unfurnished"])

age_map = {"0-5 Years": 3, "5-10 Years": 8, "10-20 Years": 15, "20+ Years": 25}

# ── Prediction Logic ────────────────────────────────────
if st.button("🔍 Analyze Investment", use_container_width=True):

    # Create the raw input dataframe
    raw_input = pd.DataFrame([{
        "Price_in_Lakhs": price,
        "Size_in_SqFt": size,
        "BHK": bhk,
        "Age_of_Property": age_map[age_cat],
        "Amenities_Count": amenities,
        "Transport_Score": transport,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "City": city,
        "Property_Type": prop_type,
        "Furnished_Status": furnished
    }])

    # One-Hot Encoding
    input_encoded = pd.get_dummies(raw_input, columns=["City", "Property_Type", "Furnished_Status"], drop_first=True)

    # Prepare specific dataframes for each model
    input_reg = input_encoded.reindex(columns=reg_columns, fill_value=0)
    input_clf = input_encoded.reindex(columns=clf_columns, fill_value=0)

    # Generate Predictions
    is_good = clf.predict(input_clf)[0]
    proba = clf.predict_proba(input_clf)[0][1]
    future_price = reg.predict(input_reg)[0]

    # ── Display Results ──────────────────────────────────
    st.divider()
    st.subheader("📊 Investment Analysis")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if is_good:
            st.success(f"### ✅ Good Buy\n**Investment Score:** {proba * 100:.1f}%")
            st.write("This property shows strong growth potential relative to its location and amenities.")
        else:
            st.error(f"### ⚠️ Caution\n**Investment Score:** {proba * 100:.1f}%")
            st.write("Consider negotiating the price or looking for properties with better connectivity/amenities.")

    with res_col2:
        st.metric(label="Estimated Value (5 Years)", value=f"₹{future_price:.2f} L",
                  delta=f"{((future_price - price) / price) * 100:.1f}% Total ROI")
        st.info(f"Calculated future value at ₹{future_price / size:.2f} per sq.ft.")

    st.caption("Powered by XGBoost + MLflow 🚀 | Models version: May 2026")