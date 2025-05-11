# scripts/train_pipeline.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger
import joblib
import yaml

from src.data.data_loader import DataLoader
from src.features.rfm_analysis import RFMAnalysis
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ChurnPredictor

def clean_column_names(df):
    """Clean column names to be XGBoost compatible"""
    df.columns = df.columns.str.replace('<lambda>', 'lambda')
    df.columns = df.columns.str.replace('[', '')
    df.columns = df.columns.str.replace(']', '')
    df.columns = df.columns.str.replace('<', '')
    df.columns = df.columns.str.replace('>', '')
    df.columns = df.columns.str.replace(' ', '_')
    return df

def main():
    """Main training pipeline"""
    
    # Create directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader()
    customers, products, transactions = data_loader.load_all_data()
    
    # Create RFM features
    logger.info("Performing RFM analysis...")
    rfm_analyzer = RFMAnalysis()
    rfm_features = rfm_analyzer.calculate_rfm(transactions)
    rfm_features = rfm_analyzer.create_rfm_features(rfm_features)
    
    # Feature engineering
    logger.info("Engineering features...")
    feature_engineer = FeatureEngineer()
    feature_matrix = feature_engineer.create_customer_features(
        customers, transactions, rfm_features
    )
    
    # Clean column names for XGBoost compatibility
    feature_matrix = clean_column_names(feature_matrix)
    
    # Save processed features
    feature_matrix.to_csv('data/processed/feature_matrix.csv', index=False)
    
    # Prepare data for modeling
    target = 'churned'
    exclude_cols = ['customer_id', 'churned', 'registration_date', 
                    'first_purchase_date', 'last_purchase_date',
                    'gender', 'location', 'customer_segment', 'payment_method_lambda',
                    'segment']
    
    feature_cols = [col for col in feature_matrix.columns if col not in exclude_cols]
    
    # Handle any missing values
    feature_matrix[feature_cols] = feature_matrix[feature_cols].fillna(0)
    
    X = feature_matrix[feature_cols]
    y = feature_matrix[target]
    
    # Train models
    logger.info("Training models...")
    churn_predictor = ChurnPredictor()
    results = churn_predictor.train_models(X, y)
    
    # Print results
    logger.info("\nModel Performance:")
    for model_name, metrics in results.items():
        logger.info(f"{model_name}: AUC = {metrics['auc']:.4f}")
    
    # Save best model
    churn_predictor.save_model('models/saved_models/best_model.pkl')
    
    # Save feature engineer
    joblib.dump(feature_engineer, 'models/saved_models/feature_engineer.pkl')
    
    # Plot feature importance
    churn_predictor.plot_feature_importance(top_n=20)
    
    # Create model performance report
    best_model_name = churn_predictor.best_model_name
    best_results = results[best_model_name]
    
    report = {
        'best_model': best_model_name,
        'auc_score': float(best_results['auc']),
        'cv_mean': float(best_results['cv_mean']),
        'cv_std': float(best_results['cv_std']),
        'best_params': best_results['best_params'],
        'feature_importance': churn_predictor.feature_importance.to_dict() if churn_predictor.feature_importance is not None else None
    }
    
    with open('models/saved_models/model_report.yaml', 'w') as f:
        yaml.dump(report, f)
    
    logger.info(f"\nTraining complete! Best model: {best_model_name}")
    logger.info(f"AUC Score: {best_results['auc']:.4f}")

if __name__ == "__main__":
    main()