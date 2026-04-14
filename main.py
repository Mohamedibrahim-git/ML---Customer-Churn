import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("_WELCOME TO MOHAMED IBRAHIM'S PROJECT_")
st.write("Machine Learning project using Streamlit")

@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

categorical_cols = df.select_dtypes(include="object").columns.drop("Churn").tolist()

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df["Churn"] = LabelEncoder().fit_transform(df["Churn"])

X = df.drop("Churn", axis=1)
y = df["Churn"].values 

model = LogisticRegression(max_iter=1000)
model.fit(X.values, y) 

st.subheader("🧾 Enter Customer Details")

input_data = {}

for col in X.columns:
    if col in categorical_cols:
        input_data[col] = st.selectbox(col, encoders[col].classes_)
    elif col == "tenure":
        input_data[col] = st.slider("Tenure (Months)", 0, 72, 12)
    elif col == "MonthlyCharges":
        input_data[col] = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    elif col == "TotalCharges":
        input_data[col] = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
    else:
        input_data[col] = float(np.mean(X[col].values))

for col in categorical_cols:
    input_data[col] = encoders[col].transform([input_data[col]])[0]

input_array = np.array(list(input_data.values())).reshape(1, -1)

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Customer is NOT likely to churn (Probability: {probability:.2%})")

st.markdown("---")
st.caption("End-to-end Machine Learning Deployment using Streamlit")
