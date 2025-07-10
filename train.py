import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Scripts.load_data import load_all_data
from Scripts.feature_engineering import generate_features
from Scripts.model import build_model, train_model

print(" Step 1: Starting training script...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(" Step 2: Loading data...")
orders, prior, train, products, aisles, departments = load_all_data()

print(" Step 3: Generating features...")
features_df = generate_features(orders, prior, products, aisles, departments)
print(" Features shape:", features_df.shape)

if features_df.empty:
    print("ERROR: Feature dataframe is empty. Check your feature_engineering.py logic.")
    exit()

print(" Step 4: Preprocessing and training model...")

feature_cols = ['up_order_count', 'up_reorder_count', 'user_total_orders',
                'avg_days_between_orders', 'times_reordered',
                'times_purchased', 'reorder_ratio']
X = features_df[feature_cols]
y = features_df['reordered']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.save"))

model = build_model(X_train_scaled.shape[1])
model, history = train_model(model, X_train_scaled, y_train, X_val_scaled, y_val)

MODEL_PATH = os.path.join(MODEL_DIR, "product_reorder_model.h5")
model.save(MODEL_PATH)

if os.path.exists(MODEL_PATH):
    print(f" Model saved at: {MODEL_PATH}")
else:
    print(" Model NOT saved.")

print(" Training complete!")
