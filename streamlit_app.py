# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yaml
import requests
import joblib
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# RFM Analysis class (copy from your existing code)
class RFMAnalysis:
    def __init__(self):
        self.segment_map = {
            'Champions': (1, 2, 1, 2),
            'Loyal_Customers': (3, 3, 1, 3),
            'Potential_Loyalists': (3, 3, 4, 5),
            'At_Risk': (4, 5, 1, 3),
            'Lost_Customers': (4, 5, 4, 5)
        }

    def calculate_rfm(self, transactions):
        """Calculate RFM metrics for each customer"""
        # Calculate recency, frequency, monetary value
        today = pd.Timestamp.now()
        
        rfm = transactions.groupby('customer_id').agg({
            'transaction_date': lambda x: (today - pd.to_datetime(x).max()).days,
            'transaction_id': 'count',
            'amount': 'sum'
        })
        
        # Rename columns
        rfm.columns = ['recency', 'frequency', 'monetary']
        
        # Create RFM segments
        rfm = self.create_rfm_features(rfm)
        
        return rfm
    
    def create_rfm_features(self, rfm):
        """Create RFM features and segments"""
        # Normalize values
        rfm['recency_norm'] = (rfm['recency'] - rfm['recency'].min()) / (rfm['recency'].max() - rfm['recency'].min())
        rfm['frequency_norm'] = (rfm['frequency'] - rfm['frequency'].min()) / (rfm['frequency'].max() - rfm['frequency'].min())
        rfm['monetary_norm'] = (rfm['monetary'] - rfm['monetary'].min()) / (rfm['monetary'].max() - rfm['monetary'].min())
        
        # R and F scores (1 is best, 5 is worst)
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=range(1, 6)).astype(int)
        rfm['f_score'] = pd.qcut(rfm['frequency'], 5, labels=range(1, 6)).astype(int)
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=range(1, 6)).astype(int)
        
        # RFM Score
        rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
        
        # Feature Interactions
        rfm['rf_interaction'] = rfm['recency_norm'] * rfm['frequency_norm']
        rfm['fm_interaction'] = rfm['frequency_norm'] * rfm['monetary_norm']
        rfm['rm_interaction'] = rfm['recency_norm'] * rfm['monetary_norm']
        
        # Create segments
        rfm['segment'] = 'Unknown'
        
        for segment, (r_min, r_max, f_min, f_max) in self.segment_map.items():
            rfm.loc[(rfm['r_score'] >= r_min) & (rfm['r_score'] <= r_max) & 
                   (rfm['f_score'] >= f_min) & (rfm['f_score'] <= f_max), 'segment'] = segment
        
        return rfm.reset_index()

# Load sample data
@st.cache_data
def load_sample_data():
    # Generate customers
    np.random.seed(42)
    customers = pd.DataFrame({
        'customer_id': [f'CUST{i:05d}' for i in range(1000)],
        'registration_date': [datetime.now() - timedelta(days=np.random.randint(1, 730)) for _ in range(1000)],
        'age': np.random.randint(18, 70, 1000),
        'gender': np.random.choice(['M', 'F'], 1000),
        'location': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000),
        'churned': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })

    # Generate products
    products = pd.DataFrame({
        'product_id': [f'PROD{i:04d}' for i in range(100)],
        'product_name': [f'Product {i}' for i in range(100)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], 100),
        'price': np.random.uniform(10, 500, 100)
    })

    # Generate transactions
    transactions_list = []
    for i in range(5000):
        customer_idx = np.random.randint(0, len(customers))
        product_idx = np.random.randint(0, len(products))
        
        transaction = {
            'transaction_id': f'TRX{i:06d}',
            'customer_id': customers.iloc[customer_idx]['customer_id'],
            'product_id': products.iloc[product_idx]['product_id'],
            'transaction_date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
            'quantity': np.random.randint(1, 5),
            'amount': products.iloc[product_idx]['price'] * np.random.randint(1, 5),
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'])
        }
        transactions_list.append(transaction)

    transactions = pd.DataFrame(transactions_list)
    
    return customers, products, transactions

# Main dashboard
def main():
    st.title("🛍️ E-commerce Customer Analytics Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", 
                               ["Overview", "Customer Segmentation", "Churn Analysis", "Individual Prediction"])
    
    # Load data
    customers, products, transactions = load_sample_data()
    
    if page == "Overview":
        show_overview(customers, transactions)
    elif page == "Customer Segmentation":
        show_segmentation(customers, transactions)
    elif page == "Churn Analysis":
        show_churn_analysis(customers, transactions)
    elif page == "Individual Prediction":
        show_prediction_page()

def show_overview(customers, transactions):
    st.header("Business Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(customers):,}")
    
    with col2:
        st.metric("Total Revenue", f"${transactions['amount'].sum():,.2f}")
    
    with col3:
        st.metric("Avg Transaction Value", f"${transactions['amount'].mean():.2f}")
    
    with col4:
        churn_rate = customers['churned'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    # Transaction trends
    st.subheader("Transaction Trends")
    
    # Monthly revenue
    transactions['month'] = pd.to_datetime(transactions['transaction_date']).dt.to_period('M')
    monthly_revenue = transactions.groupby('month')['amount'].sum().reset_index()
    monthly_revenue['month'] = monthly_revenue['month'].astype(str)
    
    fig_revenue = px.line(monthly_revenue, x='month', y='amount', 
                         title='Monthly Revenue Trend',
                         labels={'amount': 'Revenue ($)', 'month': 'Month'})
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Customer distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_segment = px.pie(customers, names='customer_segment', 
                           title='Customer Segments',
                           color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_segment, use_container_width=True)
    
    with col2:
        location_counts = customers['location'].value_counts().reset_index()
        location_counts.columns = ['location', 'count']
        fig_location = px.bar(location_counts, x='location', y='count',
                            title='Customers by Location',
                            labels={'count': 'Number of Customers', 'location': 'Location'},
                            color='count',
                            color_continuous_scale='Blues')
        st.plotly_chart(fig_location, use_container_width=True)

def show_segmentation(customers, transactions):
    st.header("Customer Segmentation Analysis")
    
    # RFM Analysis
    rfm_analyzer = RFMAnalysis()
    rfm_df = rfm_analyzer.calculate_rfm(transactions)
    
    # RFM Distribution
    st.subheader("RFM Score Distribution")
    
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=('Recency Distribution', 
                                     'Frequency Distribution', 
                                     'Monetary Distribution'))
    
    fig.add_trace(go.Histogram(x=rfm_df['recency'], name='Recency', 
                              marker_color='indianred'), row=1, col=1)
    fig.add_trace(go.Histogram(x=rfm_df['frequency'], name='Frequency',
                              marker_color='lightseagreen'), row=1, col=2)
    fig.add_trace(go.Histogram(x=rfm_df['monetary'], name='Monetary',
                              marker_color='royalblue'), row=1, col=3)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segments
    st.subheader("Customer Segments")
    
    segment_counts = rfm_df['segment'].value_counts().reset_index()
    segment_counts.columns = ['segment', 'count']
    fig_segments = px.bar(segment_counts, x='segment', y='count',
                         title='Customers by RFM Segment',
                         labels={'count': 'Number of Customers', 'segment': 'Segment'},
                         color='count',
                         color_continuous_scale='Viridis')
    st.plotly_chart(fig_segments, use_container_width=True)
    
    # Segment metrics
    segment_metrics = rfm_df.groupby('segment').agg({
        'monetary': 'mean',
        'frequency': 'mean',
        'recency': 'mean'
    }).round(2)
    
    st.subheader("Segment Metrics")
    st.dataframe(segment_metrics)

def show_churn_analysis(customers, transactions):
    st.header("Churn Analysis")
    
    # Churn by segment
    churn_by_segment = customers.groupby('customer_segment')['churned'].agg(['mean', 'count']).reset_index()
    churn_by_segment.columns = ['customer_segment', 'churn_rate', 'customer_count']
    churn_by_segment['churn_rate'] = churn_by_segment['churn_rate'] * 100
    
    st.subheader("Churn Rate by Customer Segment")
    fig_churn_segment = px.bar(churn_by_segment, 
                              x='customer_segment', 
                              y='churn_rate',
                              color='churn_rate',
                              color_continuous_scale='RdYlBu_r',
                              title='Churn Rate by Customer Segment',
                              labels={'churn_rate': 'Churn Rate (%)', 
                                     'customer_segment': 'Customer Segment'})
    st.plotly_chart(fig_churn_segment, use_container_width=True)
    
    # Churn factors
    st.subheader("Churn Factors Analysis")
    
    # Add transaction metrics to customers
    customer_metrics = transactions.groupby('customer_id').agg({
        'amount': 'sum',
        'transaction_id': 'count',
        'transaction_date': 'max'
    }).reset_index()
    
    customers_with_metrics = customers.merge(customer_metrics, on='customer_id', how='left')
    customers_with_metrics['days_since_last_purchase'] = (
        pd.Timestamp.now() - pd.to_datetime(customers_with_metrics['transaction_date'])
    ).dt.days
    
    # Fill NaN values
    customers_with_metrics['amount'] = customers_with_metrics['amount'].fillna(0)
    customers_with_metrics['transaction_id'] = customers_with_metrics['transaction_id'].fillna(0)
    customers_with_metrics['days_since_last_purchase'] = customers_with_metrics['days_since_last_purchase'].fillna(999)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_age = px.box(customers_with_metrics, 
                        x='churned', 
                        y='age',
                        color='churned',
                        title='Age Distribution by Churn Status',
                        labels={'churned': 'Churned', 'age': 'Age'},
                        color_discrete_map={0: 'lightblue', 1: 'lightcoral'})
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        fig_spent = px.box(customers_with_metrics, 
                          x='churned', 
                          y='amount',
                          color='churned',
                          title='Total Spent by Churn Status',
                          labels={'churned': 'Churned', 'amount': 'Total Spent ($)'},
                          color_discrete_map={0: 'lightblue', 1: 'lightcoral'})
        st.plotly_chart(fig_spent, use_container_width=True)

def show_prediction_page():
    st.header("Individual Customer Churn Prediction")
    
    st.markdown("Enter customer details to predict churn probability:")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        customer_id = st.text_input("Customer ID", "CUST00001")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["M", "F"])
        location = st.selectbox("Location", ["North", "South", "East", "West"])
        customer_segment = st.selectbox("Customer Segment", ["Premium", "Standard", "Basic"])
    
    with col2:
        total_transactions = st.number_input("Total Transactions", min_value=0, value=10)
        total_spent = st.number_input("Total Spent ($)", min_value=0.0, value=500.0)
        days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, value=30)
        tenure_days = st.number_input("Tenure (Days)", min_value=0, value=365)
    
    if st.button("Predict Churn", type="primary"):
        # Sample prediction logic (normally would call API)
        churn_risk = 0.3 if age > 40 and total_transactions < 5 else 0.7 if days_since_last_purchase > 60 else 0.1
        
        # Set risk level
        if churn_risk < 0.3:
            risk_level = "Low"
        elif churn_risk < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{churn_risk:.2%}")
        
        with col2:
            st.metric("Risk Level", risk_level)
        
        with col3:
            st.metric("Predicted", "Churn" if churn_risk > 0.5 else "No Churn")
        
        # Recommendations
        st.subheader("Recommended Actions")
        
        if risk_level == "High":
            st.error("⚠️ High risk customer - Immediate intervention required!")
            st.markdown("""
            - Send personalized retention offer
            - Schedule customer success call
            - Offer loyalty program enrollment
            """)
        elif risk_level == "Medium":
            st.warning("⚡ Medium risk customer - Monitor closely")
            st.markdown("""
            - Send engagement email campaign
            - Offer product recommendations
            - Provide exclusive discounts
            """)
        else:
            st.success("✅ Low risk customer - Maintain engagement")
            st.markdown("""
            - Continue regular communication
            - Send newsletter updates
            - Encourage referrals
            """)

if __name__ == "__main__":
    main()