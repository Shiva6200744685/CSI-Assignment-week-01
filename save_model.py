import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

print("Loading data...")
train = pd.read_csv("train.csv")

print("Preprocessing data...")
# Simple preprocessing
train_clean = train.fillna(train.median(numeric_only=True)).fillna('None')
train_encoded = pd.get_dummies(train_clean)

# Split data
X = train_encoded.drop(['Id', 'SalePrice'], axis=1)
y = train_encoded['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
# Train model with best parameters from hyperparameter tuning
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("Saving model and feature information...")
feature_names = X.columns.tolist()

# Calculate feature ranges for numeric inputs
feature_ranges = {}
for col in train.select_dtypes(include=['number']).columns:
    if col not in ['Id', 'SalePrice']:
        feature_ranges[col] = {
            'min': train[col].min(),
            'max': train[col].max(),
            'mean': train[col].mean(),
            'type': 'numeric'
        }

# Add categorical features
for col in train.select_dtypes(include=['object']).columns:
    feature_ranges[col] = {
        'values': train[col].dropna().unique().tolist(),
        'type': 'categorical'
    }

# Save model and feature information
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
joblib.dump(feature_ranges, 'feature_ranges.pkl')

print("Model and feature information saved successfully!")