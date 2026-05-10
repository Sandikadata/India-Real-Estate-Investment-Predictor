import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

# ── Load Data ────────────────────────────────────
df = pd.read_csv("india_housing_prices.csv")
df = df.sample(n=20000, random_state=42).reset_index(drop=True)

# ── Feature Engineering ──────────────────────────
df['Age_of_Property'] = 2026 - df['Year_Built']
df['Amenities_Count'] = df['Amenities'].apply(lambda x: len(str(x).split(',')))
transport_map = {"Low": 0, "Medium": 1, "High": 2}
df['Transport_Score'] = df['Public_Transport_Accessibility'].map(transport_map)

# ── Target Variables ───────────────────────
city_growth = {"Mumbai": 0.10, "Pune": 0.09, "Bangalore": 0.10, "Hyderabad": 0.09,
               "Chennai": 0.08, "Delhi": 0.09, "New Delhi": 0.09, "Noida": 0.08, "Gurgaon": 0.09}
df['Growth_Rate'] = df['City'].map(city_growth).fillna(0.07)
df['Future_Price'] = df['Price_in_Lakhs'] * ((1 + df['Growth_Rate']) ** 5)

df['Investment_Score'] = (
        (df['Nearby_Schools'] * 1.5) + (df['Nearby_Hospitals'] * 1.5) +
        (df['Transport_Score'] * 3.0) + (df['Amenities_Count'] * 2.0) - (df['Age_of_Property'] * 0.5)
)
threshold = df['Investment_Score'].median()
df['Good_Investment'] = (df['Investment_Score'] > threshold).astype(int)

# ── Preprocessing & One-Hot Encoding ──────────────
features_base = ['Price_in_Lakhs', 'Size_in_SqFt', 'BHK', 'Age_of_Property',
                 'Amenities_Count', 'Transport_Score', 'Nearby_Schools', 'Nearby_Hospitals',
                 'City', 'Property_Type', 'Furnished_Status']

X_base = df[features_base].copy()
# Fixed: We specify columns to encode to match main.py logic
X_encoded = pd.get_dummies(X_base, columns=['City', 'Property_Type', 'Furnished_Status'], drop_first=True)

X_reg = X_encoded.copy()
X_clf = X_encoded.drop(columns=['Price_in_Lakhs'])

y_class = df['Good_Investment']
y_reg = df['Future_Price']

# ── Train Models ─────────────────────────────
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_class, test_size=0.2, random_state=42)

mlflow.set_experiment("Housing_Investment_Analysis")
with mlflow.start_run(run_name="Final_Model_Run"):
    reg_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    reg_model.fit(X_train_r, y_train_r)

    clf_model = XGBClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_c, y_train_c)

    # Log & Save

    joblib.dump(reg_model, "regressor.pkl", compress=3)
    joblib.dump(clf_model, "classifier.pkl", compress=3)
    joblib.dump(X_reg.columns.tolist(), "reg_columns.pkl")
    joblib.dump(X_clf.columns.tolist(), "clf_columns.pkl")
    print("✅ Training and saving complete!")