# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict
import sys
import os
import traceback

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import after adding to path
from src.features.feature_engineering import FeatureEngineer
from src.data.data_loader import DataLoader

# Create FastAPI app
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Get absolute paths for model files
model_path = os.path.join(parent_dir, "models", "saved_models", "best_model.pkl")
feature_engineer_path = os.path.join(parent_dir, "models", "saved_models", "feature_engineer.pkl")

# Check if model files exist
if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
    print("Please run train_pipeline.py first to train and save the model.")
    print("From project root directory, run: python scripts/train_pipeline.py")
    sys.exit(1)

# Load model
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    
    # Get feature names from the model if available
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        print(f"Expected features: {expected_features}")
    else:
        # Fallback to default features
        expected_features = None
        print("Warning: Model doesn't have feature_names_in_ attribute")
        
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

class CustomerData(BaseModel):
    customer_id: str
    age: int
    gender: str
    location: str
    customer_segment: str
    total_transactions: int
    total_spent: float
    days_since_last_purchase: int
    tenure_days: int

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """Predict churn for a single customer"""
    
    try:
        # Convert to DataFrame using model_dump() instead of dict()
        customer_dict = customer.model_dump()
        customer_df = pd.DataFrame([customer_dict])
        
        # If we know the expected features, create a DataFrame with all required features
        if expected_features:
            # Create empty DataFrame with all expected features
            features_df = pd.DataFrame(columns=expected_features)
            
            # Map the input features to expected feature names
            feature_mapping = {
                'age': 'age',
                'total_transactions': 'total_transactions',
                'total_spent': 'amount_sum',
                'days_since_last_purchase': 'days_since_last_purchase',
                'tenure_days': 'tenure_days'
            }
            
            # Create a row with all features, defaulting missing ones to 0
            row_data = {col: 0.0 for col in expected_features}
            
            # Fill in the features we have
            for input_col, model_col in feature_mapping.items():
                if input_col in customer_dict and model_col in expected_features:
                    row_data[model_col] = customer_dict[input_col]
            
            features_df = pd.DataFrame([row_data])
        else:
            # Fallback to simple feature selection
            feature_cols = ['age', 'total_transactions', 'total_spent', 
                           'days_since_last_purchase', 'tenure_days']
            features_df = customer_df[feature_cols]
        
        # Make prediction
        probability = model.predict_proba(features_df)[0][1]
        prediction = model.predict(features_df)[0]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=float(probability),
            churn_prediction=bool(prediction),
            risk_level=risk_level
        )
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/feature_info")
def get_feature_info():
    """Get information about model features"""
    
    if hasattr(model, 'feature_names_in_'):
        return {
            "expected_features": list(model.feature_names_in_),
            "num_features": len(model.feature_names_in_)
        }
    else:
        return {
            "message": "Feature information not available",
            "note": "Model doesn't have feature_names_in_ attribute"
        }

@app.get("/model_info")
def get_model_info():
    """Get information about the loaded model"""
    
    return {
        "model_type": type(model).__name__,
        "model_path": model_path,
        "api_version": "1.0",
        "project_root": parent_dir,
        "has_feature_names": hasattr(model, 'feature_names_in_')
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting API server...")
    print(f"Project root: {parent_dir}")
    print(f"Model path: {model_path}")
    uvicorn.run(app, host="0.0.0.0", port=8000)