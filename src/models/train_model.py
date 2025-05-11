# src/models/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any

class ChurnPredictor:
    """Class to handle model training and prediction"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple models and select the best one"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Hyperparameter tuning
            params = self._get_hyperparameters(name)
            grid_search = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train_balanced, y_train_balanced)
            
            # Best model from grid search
            best_model = grid_search.best_estimator_
            
            # Evaluate
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': best_model,
                'auc': auc_score,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'best_params': grid_search.best_params_
            }
            
            logger.info(f"{name} - AUC: {auc_score:.4f}, CV Mean: {cv_scores.mean():.4f}")
        
        # Select best model
        best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        # Calculate feature importance
        self._calculate_feature_importance(X.columns)
        
        return results
    
    def _get_hyperparameters(self, model_name: str) -> Dict:
        """Get hyperparameter grid for each model"""
        
        if model_name == 'random_forest':
            return {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif model_name == 'xgboost':
            return {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        elif model_name == 'lightgbm':
            return {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            }
        
    def _calculate_feature_importance(self, feature_names):
        """Calculate and store feature importance"""
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance"""
        
        if self.feature_importance is None:
            logger.warning("Feature importance not calculated yet")
            return
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=self.feature_importance.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, filepath: str):
        """Save the best model"""
        joblib.dump(self.best_model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        self.best_model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data"""
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)[:, 1]
        return predictions, probabilities