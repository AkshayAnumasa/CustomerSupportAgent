from pymongo import MongoClient
from datetime import datetime, timedelta
import json

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['customer_support_db']
orders_collection = db['orders']
customers_collection = db['customers']

# Sample order data
sample_orders = [
    {
        "order_id": "ORD001",
        "customer_id": "CUST001",
        "status": "delivered",
        "items": [
            {"name": "Laptop", "quantity": 1, "price": 999.99},
            {"name": "Mouse", "quantity": 1, "price": 29.99}
        ],
        "total_amount": 1029.98,
        "created_at": datetime.now() - timedelta(days=5),
        "updated_at": datetime.now() - timedelta(days=1)
    },
    {
        "order_id": "ORD002",
        "customer_id": "CUST002",
        "status": "processing",
        "items": [
            {"name": "Headphones", "quantity": 1, "price": 199.99}
        ],
        "total_amount": 199.99,
        "created_at": datetime.now() - timedelta(days=1),
        "updated_at": datetime.now()
    },
    {
        "order_id": "ORD003",
        "customer_id": "CUST001",
        "status": "pending",
        "items": [
            {"name": "Keyboard", "quantity": 1, "price": 89.99},
            {"name": "Monitor", "quantity": 2, "price": 299.99}
        ],
        "total_amount": 689.97,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
]

# Load customer data from JSON file
with open('sample_customer_data.json', 'r') as f:
    customer_data = json.load(f)

# Clean up the data (remove incomplete entries)
customer_data = [entry for entry in customer_data if 'CustomerID' in entry]

# Clear existing data and insert new data
orders_collection.delete_many({})
customers_collection.delete_many({})

# Insert customer data
customers_collection.insert_many(customer_data)

print(f"Successfully loaded {len(customer_data)} customer records into MongoDB!")

# Create indexes for better query performance
orders_collection.create_index([("OrderID", 1)], unique=True)
orders_collection.create_index([("CustomerID", 1)])
customers_collection.create_index([("CustomerID", 1)], unique=True)
customers_collection.create_index([("Email", 1)])
customers_collection.create_index([("AccountID", 1)])

print("Database setup completed successfully!")
