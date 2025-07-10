# app.py
import streamlit as st
import numpy as np
import pandas as pd
import os
from keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64

from Scripts.load_data import load_all_data
from Scripts.feature_engineering import generate_features

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "product_reorder_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.save")
BACKGROUND_PATH = os.path.join(BASE_DIR, "assets", "grocery_background.jpg")  # Add your professional image here

# -------------------------------------------------------------
# Set Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="Product Reorder Prediction",
    page_icon="🛒",
    layout="wide",
)

# -------------------------------------------------------------
# Add background image using base64
# -------------------------------------------------------------
def add_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

if os.path.exists(BACKGROUND_PATH):
    add_background(BACKGROUND_PATH)

# -------------------------------------------------------------
# Safety checks
# -------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Trained model not found at {MODEL_PATH}. Run `python train.py` first.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("Scaler not found. Run `python train.py` first.")
    st.stop()

# -------------------------------------------------------------
# Load model & scaler
# -------------------------------------------------------------
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------------------------------------
# Title
# -------------------------------------------------------------
st.markdown("""
    <h1 style='color:#154360;'> Product Reorder Prediction App</h1>
    <h4 style='color:#1B4F72;'>Explore customer-product interactions and predict reorders using a Deep Learning model.</h4>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Cache loading of feature dataframe
# -------------------------------------------------------------
@st.cache_data(show_spinner="Loading datasets & computing features…")
def load_feature_data():
    orders, prior, train, products, aisles, departments = load_all_data()
    features_df = generate_features(orders, prior, products, aisles, departments)

    features_df = features_df.merge(products[['product_id', 'product_name', 'aisle_id']], on='product_id', how='left')
    features_df = features_df.merge(aisles[['aisle_id', 'aisle']], on='aisle_id', how='left')
    user_names = {uid: f"User_{uid}" for uid in features_df['user_id'].unique()}
    features_df['user_name'] = features_df['user_id'].map(user_names)

    return features_df

features_df = load_feature_data()

# -------------------------------------------------------------
# User & product selection
# -------------------------------------------------------------
st.sidebar.header(" Select User & Product")
user_display = features_df[['user_id', 'user_name']].drop_duplicates()
user_display['display'] = user_display['user_name'] + " (ID: " + user_display['user_id'].astype(str) + ")"
user_option = st.sidebar.selectbox("Select User", user_display['display'])
selected_user = int(user_option.split("ID: ")[-1].replace(")", ""))

user_products = features_df[features_df['user_id'] == selected_user].copy()
product_display = user_products[['product_id', 'product_name']].drop_duplicates()
product_display['display'] = product_display['product_name'] + " (ID: " + product_display['product_id'].astype(str) + ")"
product_option = st.sidebar.selectbox("Select Product", product_display['display'])
selected_product = int(product_option.split("ID: ")[-1].replace(")", ""))

subset = user_products[user_products['product_id'] == selected_product]

if subset.empty:
    st.warning(" No interaction found for this user-product combination.")
    st.stop()

product_name = subset['product_name'].values[0]
aisle_name = subset['aisle'].values[0]
user_name = subset['user_name'].values[0]

st.markdown(f"""
    <div style='background-color:#F9F9F9;padding:15px;border-radius:10px'>
        <h4 style='color:#2E86C1;'>👤 User: {user_name} 📦 Product: {product_name}</h4>
        <h4 style='color:#1B4F72;'>🛒 Aisle: {aisle_name}</h4>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Top 5 reordered products
# -------------------------------------------------------------
st.markdown("""
    <div style='background-color:#EAF2F8;padding:10px;border-radius:8px'>
        <h4 style='color:#154360;'> Top 5 Reordered Products by User</h4>
    </div>
    """, unsafe_allow_html=True)
top5 = user_products.sort_values(by='up_reorder_count', ascending=False)
top5_display = top5[['product_name', 'up_reorder_count']].drop_duplicates().head(5).reset_index(drop=True)
st.dataframe(top5_display.rename(columns={'product_name': 'Product', 'up_reorder_count': 'Reorder Count'}), use_container_width=True)

# -------------------------------------------------------------
# Feature values
# -------------------------------------------------------------
feature_cols = ['up_order_count', 'up_reorder_count', 'user_total_orders',
                'avg_days_between_orders', 'times_reordered', 'times_purchased',
                'reorder_ratio']
st.markdown("""
    <h4 style='color:#1A5276;'> Auto‑filled Feature Values</h4>
    """, unsafe_allow_html=True)
#st.dataframe(subset[feature_cols], use_container_width=True)
st.dataframe(subset[feature_cols].drop_duplicates().reset_index(drop=True))
# -------------------------------------------------------------
# Threshold slider and prediction
# -------------------------------------------------------------
st.markdown("""
    <h4 style='color:#2471A3;'> Predict Reorder Probability</h4>
    """, unsafe_allow_html=True)
threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict"):
    X = subset[feature_cols].values
    X_scaled = scaler.transform(X)
    prob = model.predict(X_scaled)[0][0]

    fig, ax = plt.subplots(figsize=(6, 2))
    sns.barplot(x=["Reorder Probability"], y=[prob], color="#3498DB", ax=ax)
    ax.axhline(threshold, color='red', linestyle='--', label=f"Threshold ({threshold:.2f})")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"**Reorder Probability:** `{prob:.2%}`")
    if prob > threshold:
        st.success(" Likely to reorder")
    else:
        st.error(" Unlikely to reorder")
