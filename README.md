# House Price Prediction Web App

This project deploys a machine learning model for house price prediction as an interactive web application using Streamlit.

## Features

- **Interactive Input**: Adjust house features using sliders and input fields
- **Instant Prediction**: Get real-time price predictions
- **Visual Insights**: View feature importance and price comparisons
- **Neighborhood Analysis**: Compare with similar houses in the same area

## Setup Instructions

1. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Train and save the model**:
   ```
   python save_model.py
   ```

3. **Run the Streamlit app**:
   ```
   streamlit run app.py
   ```

4. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## How It Works

1. The app uses an XGBoost regression model trained on the Ames Housing dataset
2. User inputs are processed and encoded to match the model's expected format
3. The model predicts the house price based on the provided features
4. Visualizations help interpret the prediction and compare with similar houses

## Model Performance

- **Algorithm**: XGBoost Regression
- **Validation RMSE**: ~26,000
- **R² Score**: ~0.91

## Files

- `app.py`: Streamlit web application
- `save_model.py`: Script to train and save the model
- `requirements.txt`: Required Python packages
- `xgboost_model.pkl`: Trained model (created by save_model.py)
- `feature_names.pkl`: Feature names for prediction (created by save_model.py)
- `feature_ranges.pkl`: Feature ranges for UI controls (created by save_model.py)