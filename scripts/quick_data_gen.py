# scripts/quick_data_gen.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Create directories
os.makedirs('../data/raw', exist_ok=True)

# Generate minimal sample data
np.random.seed(42)

# Customers
customers = pd.DataFrame({
    'customer_id': [f'CUST{i:05d}' for i in range(1000)],
    'registration_date': [datetime.now() - timedelta(days=random.randint(0, 730)) for _ in range(1000)],
    'age': np.random.randint(18, 70, 1000),
    'gender': np.random.choice(['M', 'F'], 1000),
    'location': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000)
})

# Products
products = pd.DataFrame({
    'product_id': [f'PROD{i:04d}' for i in range(100)],
    'product_name': [f'Product {i}' for i in range(100)],
    'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], 100),
    'price': np.random.uniform(10, 500, 100)
})

# Transactions
transactions_list = []
for i in range(5000):
    customer = customers.iloc[random.randint(0, len(customers)-1)]
    product = products.iloc[random.randint(0, len(products)-1)]
    
    transaction = {
        'transaction_id': f'TRX{i:06d}',
        'customer_id': customer['customer_id'],
        'product_id': product['product_id'],
        'transaction_date': datetime.now() - timedelta(days=random.randint(0, 365)),
        'quantity': random.randint(1, 5),
        'amount': product['price'] * random.randint(1, 5),
        'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'])
    }
    transactions_list.append(transaction)

transactions = pd.DataFrame(transactions_list)

# Add churn labels
last_purchase = transactions.groupby('customer_id')['transaction_date'].max()
churned_customers = last_purchase[last_purchase < (datetime.now() - timedelta(days=90))].index
customers['churned'] = customers['customer_id'].isin(churned_customers).astype(int)

# Save data
customers.to_csv('../data/raw/customers.csv', index=False)
products.to_csv('../data/raw/products.csv', index=False)
transactions.to_csv('../data/raw/transactions.csv', index=False)

print("Sample data generated successfully!")
print(f"- {len(customers)} customers")
print(f"- {len(products)} products")
print(f"- {len(transactions)} transactions")