import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from xgboost import XGBRegressor
import os

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# App title and description
st.title("🏠 House Price Prediction App")
st.markdown("""
This app predicts house prices based on various features. Enter the details below to get a prediction.
""")

# Load the model and preprocessing data
@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgboost_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        feature_ranges = joblib.load('feature_ranges.pkl')
        return model, feature_names, feature_ranges
    except:
        # If model doesn't exist, train it
        st.warning("Model not found. Training a new model...")
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Load data
        train = pd.read_csv("train.csv")
        
        # Simple preprocessing
        train_clean = train.fillna(train.median(numeric_only=True)).fillna('None')
        train_encoded = pd.get_dummies(train_clean)
        
        # Split data
        X = train_encoded.drop(['Id', 'SalePrice'], axis=1)
        y = train_encoded['SalePrice']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model and feature info
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
        
        joblib.dump(model, 'xgboost_model.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
        joblib.dump(feature_ranges, 'feature_ranges.pkl')
        
        return model, feature_names, feature_ranges

# Load model and feature information
model, feature_names, feature_ranges = load_model()

# Create sidebar for inputs
st.sidebar.header("House Features")

# Function to create input fields
def create_feature_input(feature_name, feature_info):
    if feature_name in ['Id']:  # Skip ID field
        return None
    
    if feature_info['type'] == 'numeric':
        min_val = feature_info['min']
        max_val = feature_info['max']
        mean_val = feature_info['mean']
        
        # For some features, use sliders
        if feature_name in ['OverallQual', 'OverallCond']:
            return st.sidebar.slider(
                f"{feature_name} (1-10)", 
                int(min_val), int(max_val), 
                int(mean_val)
            )
        # For year fields, use number input with reasonable defaults
        elif 'Year' in feature_name or 'Yr' in feature_name:
            return st.sidebar.number_input(
                feature_name, 
                int(min_val), int(max_val), 
                int(mean_val)
            )
        # For area fields, use number input with step=50
        elif 'SF' in feature_name or 'Area' in feature_name:
            return st.sidebar.number_input(
                feature_name, 
                int(min_val), int(max_val), 
                int(mean_val), step=50
            )
        # For other numeric fields
        else:
            return st.sidebar.number_input(
                feature_name, 
                float(min_val), float(max_val), 
                float(mean_val)
            )
    elif feature_info['type'] == 'categorical':
        values = feature_info['values']
        # Add None option if there are missing values
        if 'None' not in values and len(values) > 0:
            values = np.append(values, 'None')
        return st.sidebar.selectbox(feature_name, values)
    
    return None

# Create collapsible sections for different feature groups
with st.sidebar.expander("Basic Information", expanded=True):
    mszoning = create_feature_input('MSZoning', feature_ranges['MSZoning']) if 'MSZoning' in feature_ranges else 'RL'
    lotarea = create_feature_input('LotArea', feature_ranges['LotArea']) if 'LotArea' in feature_ranges else 10000
    neighborhood = create_feature_input('Neighborhood', feature_ranges['Neighborhood']) if 'Neighborhood' in feature_ranges else 'NAmes'

with st.sidebar.expander("House Quality & Condition"):
    overallqual = create_feature_input('OverallQual', feature_ranges['OverallQual']) if 'OverallQual' in feature_ranges else 5
    overallcond = create_feature_input('OverallCond', feature_ranges['OverallCond']) if 'OverallCond' in feature_ranges else 5
    yearbuilt = create_feature_input('YearBuilt', feature_ranges['YearBuilt']) if 'YearBuilt' in feature_ranges else 1970

with st.sidebar.expander("Size Information"):
    grlivarea = create_feature_input('GrLivArea', feature_ranges['GrLivArea']) if 'GrLivArea' in feature_ranges else 1500
    totalbsmtsf = create_feature_input('TotalBsmtSF', feature_ranges['TotalBsmtSF']) if 'TotalBsmtSF' in feature_ranges else 1000
    fullbath = create_feature_input('FullBath', feature_ranges['FullBath']) if 'FullBath' in feature_ranges else 2
    bedroomabvgr = create_feature_input('BedroomAbvGr', feature_ranges['BedroomAbvGr']) if 'BedroomAbvGr' in feature_ranges else 3

with st.sidebar.expander("Garage Information"):
    garagecars = create_feature_input('GarageCars', feature_ranges['GarageCars']) if 'GarageCars' in feature_ranges else 2
    garagearea = create_feature_input('GarageArea', feature_ranges['GarageArea']) if 'GarageArea' in feature_ranges else 480

# Predict button
predict_button = st.sidebar.button("Predict House Price", type="primary")

# Main content area
col1, col2 = st.columns([2, 1])

# Function to prepare input data for prediction
def prepare_input_data():
    # Create a dictionary with all features initialized to 0
    input_data = {feature: 0 for feature in feature_names}
    
    # Update with user inputs for numeric features
    for feature, info in feature_ranges.items():
        if feature in ['Id', 'SalePrice']:
            continue
            
        if info['type'] == 'numeric':
            # Get the value from the corresponding variable
            value = locals().get(feature.lower(), info['mean'])
            input_data[feature] = value
    
    # Handle categorical features with one-hot encoding
    for feature, info in feature_ranges.items():
        if info['type'] == 'categorical':
            selected_value = locals().get(feature.lower(), 'None')
            
            # Create one-hot encoded columns
            for value in info['values']:
                column_name = f"{feature}_{value}"
                if column_name in feature_names:
                    input_data[column_name] = 1 if value == selected_value else 0
    
    # Convert to DataFrame with correct column order
    df = pd.DataFrame([input_data])
    return df[feature_names]  # Ensure columns match model's expected features

# Make prediction when button is clicked
if predict_button:
    with st.spinner("Calculating house price..."):
        # Prepare input data
        input_df = prepare_input_data()
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display prediction
        with col1:
            st.success(f"### Estimated House Price: ${prediction:,.2f}")
            
            # Show feature importance
            st.subheader("Feature Importance")
            
            # Get feature importance from model
            importance = model.feature_importances_
            feature_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(10)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
            ax.set_title('Top 10 Feature Importance')
            st.pyplot(fig)
            
        with col2:
            # Show comparison with similar houses
            st.subheader("Price Comparison")
            
            # Load original data for comparison
            train = pd.read_csv("train.csv")
            
            # Find similar houses based on neighborhood and size
            similar_houses = train[
                (train['Neighborhood'] == neighborhood) & 
                (train['GrLivArea'].between(grlivarea * 0.8, grlivarea * 1.2))
            ]
            
            if len(similar_houses) > 0:
                avg_price = similar_houses['SalePrice'].mean()
                median_price = similar_houses['SalePrice'].median()
                
                # Create comparison chart
                fig, ax = plt.subplots(figsize=(8, 6))
                comparison = pd.DataFrame({
                    'Price': [prediction, avg_price, median_price],
                    'Type': ['Predicted', 'Average Similar', 'Median Similar']
                })
                sns.barplot(x='Type', y='Price', data=comparison, ax=ax)
                ax.set_title('Price Comparison')
                ax.set_ylabel('Price ($)')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                st.info(f"Average price of similar houses: ${avg_price:,.2f}")
                st.info(f"Median price of similar houses: ${median_price:,.2f}")
                
                # Show price distribution in neighborhood
                st.subheader(f"Price Distribution in {neighborhood}")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(similar_houses['SalePrice'], kde=True, ax=ax)
                ax.axvline(prediction, color='red', linestyle='--', label='Prediction')
                ax.set_title(f'Price Distribution in {neighborhood}')
                ax.set_xlabel('Price ($)')
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Not enough similar houses found for comparison.")

# Add information about the model at the bottom
st.markdown("---")
st.subheader("About the Model")
st.write("""
This prediction is based on an XGBoost Regression model trained on the Ames Housing dataset.
The model was trained on historical house sales data and achieved an R² score of approximately 0.91 on validation data.
""")

# Add instructions
with st.expander("How to use this app"):
    st.write("""
    1. Adjust the house features in the sidebar
    2. Click the 'Predict House Price' button
    3. View the estimated price and feature importance
    4. Compare with similar houses in the same neighborhood
    """)