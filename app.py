import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from google.colab import drive


BASE_PATH = "/content/drive/MyDrive/ml_projects/DL_ANN_Binary_classification_project"

# --- Cache heavy resources ---
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        f"{BASE_PATH}/artifacts/models/best_model.h5"
    )

    with open(f"{BASE_PATH}/label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open(f"{BASE_PATH}/onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open(f"{BASE_PATH}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# --- Streamlit UI ---
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# --- Prepare input ---
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
)

input_data = pd.concat([input_data, geo_df], axis=1)
input_scaled = scaler.transform(input_data)

# --- Prediction ---
prediction_proba = float(model.predict(input_scaled)[0][0])

st.metric("Churn Probability", f"{prediction_proba:.2%}")

if prediction_proba >= 0.5:
    st.error("Customer is likely to churn")
else:
    st.success("Customer is not likely to churn")