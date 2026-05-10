import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np

st.set_page_config(page_title="Real Estate Investment Predictor", page_icon="🏠", layout="centered")


# ── Load Assets ──
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    clf = joblib.load(os.path.join(base_path, "classifier.pkl"))
    reg = joblib.load(os.path.join(base_path, "regressor.pkl"))
    clf_columns = joblib.load(os.path.join(base_path, "clf_columns.pkl"))
    reg_columns = joblib.load(os.path.join(base_path, "reg_columns.pkl"))
    return clf, reg, clf_columns, reg_columns


try:
    clf, reg, clf_columns, reg_columns = load_assets()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ── UI Layout ──
st.title("🏠 Real Estate Investment Predictor")
st.markdown("Enter property details to evaluate investment potential and future valuation.")

st.subheader("📋 Property Details")
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Current Price (Lakhs)", 10, 1000, 50)
    size = st.slider("Size (SqFt)", 300, 10000, 1200)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5, 6])
    age_cat = st.selectbox("Property Age", ["0-5 Years", "5-10 Years", "10-20 Years", "20+ Years"])
    amenities = st.slider("Amenities Count", 1, 15, 5)

with col2:
    transport = st.slider("Transport Score (0:Low - 2:High)", 0, 2, 1)
    schools = st.number_input("Nearby Schools", 0, 50, 5)
    hospitals = st.number_input("Nearby Hospitals", 0, 50, 3)
    city = st.selectbox("City", sorted(
        ["Ahmedabad", "Amritsar", "Bangalore", "Bhopal", "Bhubaneswar", "Bilaspur", "Chennai", "Coimbatore", "Cuttack",
         "Dehradun", "Durgapur", "Dwarka", "Faridabad", "Gaya", "Gurgaon", "Guwahati", "Haridwar", "Hyderabad",
         "Indore", "Jaipur", "Jamshedpur", "Jodhpur", "Kochi", "Kolkata", "Lucknow", "Ludhiana", "Mangalore", "Mumbai",
         "Mysore", "Nagpur", "New Delhi", "Noida", "Patna", "Pune", "Raipur", "Ranchi", "Silchar", "Surat",
         "Trivandrum", "Vijayawada", "Vishakhapatnam", "Warangal"]))
    prop_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
    furnished = st.selectbox("Furnished Status", ["Furnished", "Semi-furnished", "Unfurnished"])

age_map = {"0-5 Years": 3, "5-10 Years": 8, "10-20 Years": 15, "20+ Years": 25}

# ── Prediction Logic (Merged & Cleaned) ──
if st.button("🔍 Analyze Investment", use_container_width=True):
    # 1. Create Input DataFrame
    raw_input = pd.DataFrame([{
        "Price_in_Lakhs": price, "Size_in_SqFt": size, "BHK": bhk,
        "Age_of_Property": age_map[age_cat], "Amenities_Count": amenities,
        "Transport_Score": transport, "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals, "City": city,
        "Property_Type": prop_type, "Furnished_Status": furnished
    }])

    # 2. Encoding (Matching model.py)
    input_encoded = pd.get_dummies(raw_input, columns=['City', 'Property_Type', 'Furnished_Status'], drop_first=True)

    # 3. Align Columns using the saved column lists
    input_reg = input_encoded.reindex(columns=reg_columns, fill_value=0)
    input_clf = input_encoded.reindex(columns=clf_columns, fill_value=0)

    # 4. Generate Predictions
    is_good = clf.predict(input_clf)[0]
    proba = clf.predict_proba(input_clf)[0][1]
    future_price = reg.predict(input_reg)[0]

    # ── Display Results ──
    st.divider()
    st.subheader("📊 Investment Analysis")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if is_good:
            st.success(f"### ✅ Good Buy\n**Investment Score:** {proba * 100:.1f}%")
            st.write("This property shows strong growth potential relative to its location and amenities.")
        else:
            st.error(f"### ⚠️ Caution\n**Investment Score:** {proba * 100:.1f}%")
            st.write("Consider negotiating the price or looking for properties with better connectivity.")

    with res_col2:
        roi = ((future_price - price) / price) * 100
        st.metric(label="Estimated Value (5 Years)", value=f"₹{future_price:.2f} L", delta=f"{roi:.1f}% Total ROI")
        st.info(f"Calculated future value at ₹{future_price / size:.2f} per sq.ft.")

    st.caption("Powered by XGBoost + MLflow 🚀 | Models version: May 2026")