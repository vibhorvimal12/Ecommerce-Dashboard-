# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os
import yaml
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.features.rfm_analysis import RFMAnalysis

# Set page config
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom data loader to handle config issues
class SimpleDataLoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
    
    def load_all_data(self):
        customers = pd.read_csv(self.base_path / "data" / "raw" / "customers.csv")
        products = pd.read_csv(self.base_path / "data" / "raw" / "products.csv")
        transactions = pd.read_csv(self.base_path / "data" / "raw" / "transactions.csv")
        
        # Convert date columns
        customers['registration_date'] = pd.to_datetime(customers['registration_date'])
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        
        return customers, products, transactions

# Load data with proper error handling
@st.cache_data
def load_data():
    try:
        # Try to use the original DataLoader
        from src.data.data_loader import DataLoader
        config_path = os.path.join(parent_dir, "config", "config.yaml")
        
        if os.path.exists(config_path):
            data_loader = DataLoader(config_path)
            return data_loader.load_all_data()
        else:
            # Fallback to simple loader
            st.warning("Config file not found. Using default data paths.")
            data_loader = SimpleDataLoader(parent_dir)
            return data_loader.load_all_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Final fallback - direct loading
        try:
            data_loader = SimpleDataLoader(parent_dir)
            return data_loader.load_all_data()
        except Exception as e2:
            st.error(f"Failed to load data: {str(e2)}")
            st.stop()

# Main dashboard
def main():
    st.title("🛍️ E-commerce Customer Analytics Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", 
                               ["Overview", "Customer Segmentation", "Churn Analysis", "Individual Prediction"])
    
    # Load data
    try:
        customers, products, transactions = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure you've generated the sample data by running: `python scripts/generate_sample_data.py`")
        st.stop()
    
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
    try:
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
        st.dataframe(segment_metrics.style.highlight_max(axis=0))
        
    except Exception as e:
        st.error(f"Error in RFM analysis: {str(e)}")

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
        # Check if API is running
        import requests
        
        try:
            # Make prediction request to API
            api_url = "http://localhost:8000/predict"
            
            customer_data = {
                "customer_id": customer_id,
                "age": age,
                "gender": gender,
                "location": location,
                "customer_segment": customer_segment,
                "total_transactions": total_transactions,
                "total_spent": total_spent,
                "days_since_last_purchase": days_since_last_purchase,
                "tenure_days": tenure_days
            }
            
            response = requests.post(api_url, json=customer_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
                
                with col2:
                    st.metric("Risk Level", result['risk_level'])
                
                with col3:
                    st.metric("Predicted", "Churn" if result['churn_prediction'] else "No Churn")
                
                # Recommendations based on risk level
                st.subheader("Recommended Actions")
                
                if result['risk_level'] == "High":
                    st.error("⚠️ High risk customer - Immediate intervention required!")
                    st.markdown("""
                    - Send personalized retention offer
                    - Schedule customer success call
                    - Offer loyalty program enrollment
                    """)
                elif result['risk_level'] == "Medium":
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
            else:
                st.error(f"API Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Please make sure the API is running at http://localhost:8000")
            st.info("Start the API by running: `python api/main.py` in another terminal")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()