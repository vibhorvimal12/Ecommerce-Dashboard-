# src/features/rfm_analysis.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class RFMAnalysis:
    """Perform RFM (Recency, Frequency, Monetary) analysis"""
    
    def __init__(self, reference_date=None):
        self.reference_date = reference_date or datetime.now()
        
    def calculate_rfm(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM metrics for each customer"""
        
        # Calculate RFM metrics
        rfm = transactions.groupby('customer_id').agg({
            'transaction_date': lambda x: (self.reference_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'amount': 'sum'  # Monetary
        })
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        
        # Calculate RFM scores (1-5 scale)
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
        
        # Convert to numeric
        rfm['r_score'] = rfm['r_score'].astype(int)
        rfm['f_score'] = rfm['f_score'].astype(int)
        rfm['m_score'] = rfm['m_score'].astype(int)
        
        # Calculate combined RFM score
        rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
        
        # Create RFM segments
        rfm['segment'] = rfm.apply(self._get_segment, axis=1)
        
        return rfm
    
    def _get_segment(self, row):
        """Assign customer segments based on RFM scores"""
        if row['rfm_score'] >= 12:
            return 'Champions'
        elif row['rfm_score'] >= 9:
            return 'Loyal Customers'
        elif row['rfm_score'] >= 7:
            return 'Potential Loyalists'
        elif row['rfm_score'] >= 5:
            return 'At Risk'
        else:
            return 'Lost Customers'
    
    def create_rfm_features(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from RFM analysis"""
        
        features = rfm_df.copy()
        
        # Normalize RFM values
        scaler = StandardScaler()
        features[['recency_norm', 'frequency_norm', 'monetary_norm']] = scaler.fit_transform(
            features[['recency', 'frequency', 'monetary']]
        )
        
        # Create interaction features
        features['rf_interaction'] = features['r_score'] * features['f_score']
        features['fm_interaction'] = features['f_score'] * features['m_score']
        features['rm_interaction'] = features['r_score'] * features['m_score']
        
        # Create segment dummies
        segment_dummies = pd.get_dummies(features['segment'], prefix='segment')
        features = pd.concat([features, segment_dummies], axis=1)
        
        return features