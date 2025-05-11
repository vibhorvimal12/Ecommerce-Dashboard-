
# scripts/create_config.py
import os
import yaml

def create_config():
    config = {
        'data': {
            'raw_path': 'data/raw',
            'processed_path': 'data/processed'
        },
        'model': {
            'random_seed': 42,
            'test_size': 0.2,
            'cv_folds': 5
        },
        'features': {
            'numeric_features': [
                'age', 'total_transactions', 'total_spent', 
                'avg_transaction_value', 'days_since_last_purchase',
                'tenure_days', 'recency', 'frequency', 'monetary'
            ],
            'categorical_features': [
                'gender', 'location', 'customer_segment', 'payment_method'
            ],
            'target': 'churned'
        },
        'training': {
            'models': ['random_forest', 'xgboost', 'lightgbm'],
            'metric': 'roc_auc'
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000
        },
        'dashboard': {
            'title': 'Customer Analytics Dashboard',
            'theme': 'plotly_white'
        }
    }
    
    # Create config directory if it doesn't exist
    os.makedirs('../config', exist_ok=True)
    
    # Write config file
    with open('../config/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Config file created successfully!")

if __name__ == "__main__":
    create_config()