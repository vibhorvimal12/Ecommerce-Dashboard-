# src/data/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import yaml
from loguru import logger

class DataLoader:
    """Class to handle data loading operations"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config['data']['raw_path'])
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all datasets"""
        logger.info("Loading data...")
        
        customers = pd.read_csv(self.data_path / 'customers.csv')
        products = pd.read_csv(self.data_path / 'products.csv')
        transactions = pd.read_csv(self.data_path / 'transactions.csv')
        
        # Convert date columns
        customers['registration_date'] = pd.to_datetime(customers['registration_date'])
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        
        logger.info(f"Loaded {len(customers)} customers, {len(products)} products, {len(transactions)} transactions")
        
        return customers, products, transactions
    
    def create_customer_features(self, customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create basic customer features from transaction data"""
        
        # Calculate transaction metrics per customer
        customer_metrics = transactions.groupby('customer_id').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean', 'std'],
            'transaction_date': ['min', 'max'],
            'product_id': 'nunique'
        }).round(2)
        
        # Flatten column names
        customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns.values]
        customer_metrics = customer_metrics.rename(columns={
            'transaction_id_count': 'total_transactions',
            'amount_sum': 'total_spent',
            'amount_mean': 'avg_transaction_value',
            'amount_std': 'transaction_value_std',
            'transaction_date_min': 'first_purchase_date',
            'transaction_date_max': 'last_purchase_date',
            'product_id_nunique': 'unique_products_purchased'
        })
        
        # Calculate days since last purchase
        customer_metrics['days_since_last_purchase'] = (
            pd.Timestamp.now() - customer_metrics['last_purchase_date']
        ).dt.days
        
        # Merge with customer data
        customer_features = customers.merge(customer_metrics, on='customer_id', how='left')
        
        # Fill missing values
        customer_features['total_transactions'] = customer_features['total_transactions'].fillna(0)
        customer_features['total_spent'] = customer_features['total_spent'].fillna(0)
        customer_features['days_since_last_purchase'] = customer_features['days_since_last_purchase'].fillna(999)
        
        return customer_features