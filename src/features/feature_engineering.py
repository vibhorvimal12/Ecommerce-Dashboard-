# src/features/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple

class FeatureEngineer:
    """Class to handle feature engineering operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def create_customer_features(self, customers: pd.DataFrame, transactions: pd.DataFrame, rfm_features: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive customer features"""
        
        # Transaction-based features
        transaction_features = self._create_transaction_features(transactions)
        
        # Time-based features
        time_features = self._create_time_features(customers, transactions)
        
        # Product preference features
        product_features = self._create_product_features(transactions)
        
        # Merge all features
        features = customers.merge(transaction_features, on='customer_id', how='left')
        features = features.merge(time_features, on='customer_id', how='left')
        features = features.merge(product_features, on='customer_id', how='left')
        features = features.merge(rfm_features, on='customer_id', how='left')
        
        # Create derived features
        features = self._create_derived_features(features)
        
        # Handle categorical variables
        features = self._encode_categorical_features(features)
        
        # Fill missing values
        features = self._handle_missing_values(features)
        
        return features
    
    def _create_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create transaction-based features"""
        
        features = transactions.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'std', 'min', 'max'],
            'quantity': ['sum', 'mean'],
            'transaction_id': 'count',
            'payment_method': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        })
        
        features.columns = ['_'.join(col).strip() for col in features.columns.values]
        
        # Calculate transaction frequency
        first_last = transactions.groupby('customer_id')['transaction_date'].agg(['min', 'max'])
        features['days_active'] = (first_last['max'] - first_last['min']).dt.days
        features['transaction_frequency'] = features['transaction_id_count'] / (features['days_active'] + 1)
        
        return features.reset_index()
    
    def _create_time_features(self, customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        features = pd.DataFrame()
        features['customer_id'] = customers['customer_id']
        
        # Customer tenure
        features['tenure_days'] = (pd.Timestamp.now() - customers['registration_date']).dt.days
        
        # Transaction patterns
        trans_time = transactions.copy()
        trans_time['hour'] = trans_time['transaction_date'].dt.hour
        trans_time['day_of_week'] = trans_time['transaction_date'].dt.dayofweek
        trans_time['is_weekend'] = trans_time['day_of_week'].isin([5, 6]).astype(int)
        
        time_patterns = trans_time.groupby('customer_id').agg({
            'hour': 'mean',
            'is_weekend': 'mean'
        }).rename(columns={'hour': 'avg_transaction_hour', 'is_weekend': 'weekend_transaction_ratio'})
        
        features = features.merge(time_patterns, on='customer_id', how='left')
        
        return features
    
    def _create_product_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create product preference features"""
        
        # Get product categories (would normally join with products table)
        # For simplicity, creating random categories
        categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
        transactions['category'] = np.random.choice(categories, size=len(transactions))
        
        # Category preferences
        category_features = pd.crosstab(transactions['customer_id'], transactions['category'], normalize='index')
        category_features.columns = [f'category_{cat.lower()}_ratio' for cat in category_features.columns]
        
        # Product diversity
        product_diversity = transactions.groupby('customer_id')['product_id'].nunique().rename('product_diversity')
        
        features = pd.concat([category_features, product_diversity], axis=1).reset_index()
        
        return features
    
    def _create_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones"""
        
        # Customer lifetime value proxy
        features['customer_lifetime_value'] = features['amount_sum'] * features['transaction_frequency']
        
        # Engagement score
        features['engagement_score'] = (
            features['transaction_id_count'] * 0.3 +
            features['product_diversity'] * 0.3 +
            features['amount_sum'] / 1000 * 0.4
        )
        
        # Risk indicators
        features['high_recency_risk'] = (features['recency'] > features['recency'].quantile(0.75)).astype(int)
        features['low_frequency_risk'] = (features['frequency'] < features['frequency'].quantile(0.25)).astype(int)
        
        return features
    
    def _encode_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        
        categorical_cols = ['gender', 'location', 'customer_segment', 'payment_method_<lambda>']
        
        for col in categorical_cols:
            if col in features.columns:
                le = LabelEncoder()
                features[f'{col}_encoded'] = le.fit_transform(features[col].fillna('Unknown'))
                self.label_encoders[col] = le
        
        return features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        
        # Fill numeric columns with 0 or median
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if features[col].isnull().sum() > 0:
                if 'ratio' in col or 'score' in col:
                    features[col] = features[col].fillna(0)
                else:
                    features[col] = features[col].fillna(features[col].median())
        
        # Fill categorical columns
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            features[col] = features[col].fillna('Unknown')
        
        return features
    
    def get_feature_names(self, exclude_cols: List[str] = None) -> List[str]:
        """Get list of feature names for modeling"""
        
        exclude_cols = exclude_cols or ['customer_id', 'churned', 'registration_date', 
                                        'first_purchase_date', 'last_purchase_date']
        
        return [col for col in self.features.columns if col not in exclude_cols]