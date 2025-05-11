import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_sample_data(n_customers=10000, n_transactions=100000):
    """Generate sample e-commerce data for demonstration"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs('../data/raw', exist_ok=True)
    
    # Generate customer data
    customers = []
    for i in range(n_customers):
        customer = {
            'customer_id': f'CUST{i+1:05d}',
            'registration_date': datetime.now() - timedelta(days=random.randint(0, 730)),
            'age': random.randint(18, 70),
            'gender': random.choice(['M', 'F']),
            'location': random.choice(['North', 'South', 'East', 'West']),
            'customer_segment': random.choice(['Premium', 'Standard', 'Basic'])
        }
        customers.append(customer)
    
    customers_df = pd.DataFrame(customers)
    
    # Generate product data
    product_categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
    products = []
    for i in range(500):
        product = {
            'product_id': f'PROD{i+1:04d}',
            'product_name': f'Product {i+1}',
            'category': random.choice(product_categories),
            'price': round(random.uniform(10, 500), 2)
        }
        products.append(product)
    
    products_df = pd.DataFrame(products)
    
    # Generate transaction data
    transactions = []
    for i in range(n_transactions):
        customer = random.choice(customers)
        product = random.choice(products)
        
        # Create temporal patterns (some customers are more likely to churn)
        days_since_registration = (datetime.now() - customer['registration_date']).days
        
        # Simulate churn behavior
        if customer['customer_segment'] == 'Basic' and days_since_registration > 365:
            if random.random() > 0.7:  # 70% chance to skip transaction
                continue
        
        transaction = {
            'transaction_id': f'TRX{i+1:06d}',
            'customer_id': customer['customer_id'],
            'product_id': product['product_id'],
            'transaction_date': datetime.now() - timedelta(days=random.randint(0, 365)),
            'quantity': random.randint(1, 5),
            'amount': product['price'] * random.randint(1, 5),
            'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'])
        }
        transactions.append(transaction)
    
    transactions_df = pd.DataFrame(transactions)
    
    # Add churn labels (customers who haven't purchased in last 90 days)
    last_purchase = transactions_df.groupby('customer_id')['transaction_date'].max()
    churned_customers = last_purchase[last_purchase < (datetime.now() - timedelta(days=90))].index
    
    customers_df['churned'] = customers_df['customer_id'].isin(churned_customers).astype(int)
    
    # Save data
    customers_df.to_csv('../data/raw/customers.csv', index=False)
    products_df.to_csv('../data/raw/products.csv', index=False)
    transactions_df.to_csv('../data/raw/transactions.csv', index=False)
    
    print(f"Generated sample data:")
    print(f"- {len(customers_df)} customers")
    print(f"- {len(products_df)} products")
    print(f"- {len(transactions_df)} transactions")
    print(f"- {customers_df['churned'].sum()} churned customers")

if __name__ == "__main__":
    generate_sample_data()