import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from groq import Groq
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from datetime import datetime
import json
import re

# Mock Database using dictionaries instead of MongoDB
class MockCollection:
    def __init__(self, initial_data=None):
        self.data = initial_data or []
        
    def find_one(self, query):
        for item in self.data:
            match = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                return item
        return None
    
    def find(self, query):
        results = []
        for item in self.data:
            match = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                results.append(item)
        return results
    
    def update_one(self, query, update):
        for i, item in enumerate(self.data):
            match = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                for key, value in update["$set"].items():
                    self.data[i][key] = value
                return type('obj', (object,), {'modified_count': 1})
        return type('obj', (object,), {'modified_count': 0})

# Load sample data from JSON file
try:
    with open('sample_customer_data.json', 'r') as f:
        sample_data = json.load(f)
        # The JSON file contains combined customer and order data
        # We'll separate them for our mock database
        customers_data = []
        orders_data = []
        
        # Extract customer and order information from each record
        for record in sample_data:
            # Extract customer information
            customer = {
                "CustomerID": record.get("CustomerID", ""),
                "Name": record.get("Name", ""),
                "Email": record.get("Email", ""),
                "Contact": record.get("Contact", ""),
                "AccountID": record.get("AccountID", "")
            }
            
            # Extract order information
            order = {
                "OrderID": record.get("OrderID", ""),
                "CustomerID": record.get("CustomerID", ""),
                "status": record.get("Status", "").lower(),
                "delivery": record.get("Delivery", ""),
                "quantity": record.get("Quantity", 0),
                "invoice": record.get("Invoice", ""),
                "payment_status": record.get("Payment", ""),
                "shipping_address": record.get("ShippingAddress", ""),
                "feedback": record.get("Feedback", ""),
                "delivery_history": record.get("DeliveryHistory", []),
                "pending_action": record.get("PendingAction", ""),
                "created_at": "2025-06-01",  # Default date
                "updated_at": "2025-06-05"   # Default date
            }
            
            # Add to our collections if the record contains the required fields
            if record.get("CustomerID") and record.get("OrderID"):
                customers_data.append(customer)
                orders_data.append(order)
        
    st.sidebar.success(f"Loaded {len(orders_data)} orders from sample data")
    
except Exception as e:
    st.sidebar.warning(f"Failed to load sample data: {e}. Using default mock data.")
    # Default mock data if file can't be loaded
    orders_data = [
        {
            "OrderID": "12345",
            "CustomerID": "CUST001",
            "status": "delivered",
            "items": [
                {"name": "Laptop", "quantity": 1, "price": 999.99},
                {"name": "Mouse", "quantity": 1, "price": 29.99}
            ],
            "total_amount": 1029.98,
            "created_at": "2025-06-01",
            "updated_at": "2025-06-03"
        },
        {
            "OrderID": "67890",
            "CustomerID": "CUST002",
            "status": "processing",
            "items": [
                {"name": "Headphones", "quantity": 1, "price": 199.99}
            ],
            "total_amount": 199.99,
            "created_at": "2025-06-05",
            "updated_at": "2025-06-05"
        }
    ]

    customers_data = [
        {
            "CustomerID": "CUST001",
            "Name": "John Smith",
            "Email": "john@example.com",
            "Contact": "555-123-4567",
            "AccountID": "ACC001"
        },
        {
            "CustomerID": "CUST002",
            "Name": "Jane Doe",
            "Email": "jane@example.com",
            "Contact": "555-987-6543",
            "AccountID": "ACC002"
        }
    ]

# Initialize mock collections
orders_collection = MockCollection(orders_data)
customers_collection = MockCollection(customers_data)

st.sidebar.success("Using local mock database")

# Customer Management Functions
def find_customer(customer_id=None, email=None, account_id=None):
    """Find customer by ID, email, or account ID"""
    try:
        if customer_id:
            return customers_collection.find_one({"CustomerID": customer_id})
        elif email:
            return customers_collection.find_one({"Email": email})
        elif account_id:
            return customers_collection.find_one({"AccountID": account_id})
    except Exception as e:
        st.error(f"Error finding customer: {e}")
    return None

def find_customer_orders(customer_id):
    """Find all orders for a customer"""
    try:
        return list(orders_collection.find({"CustomerID": customer_id}))
    except Exception as e:
        st.error(f"Error finding customer orders: {e}")
        return []

# Order Management Functions
def find_order(order_id):
    """Find order by ID"""
    try:
        # First try exact match
        order = orders_collection.find_one({"OrderID": order_id})
        
        # If not found, try case-insensitive match
        if not order:
            for item in orders_collection.data:
                if item.get("OrderID", "").lower() == order_id.lower():
                    order = item
                    break
        
        if order:
            # Get customer details
            customer = find_customer(customer_id=order.get("CustomerID"))
            if customer:
                order["customer_details"] = {
                    "Name": customer.get("Name"),
                    "Email": customer.get("Email"),
                    "Contact": customer.get("Contact")
                }
        return order
    except Exception as e:
        st.error(f"Error finding order: {e}")
        return None

def find_orders_by_customer(customer_id):
    """Find all orders for a customer"""
    try:
        orders = list(orders_collection.find({"CustomerID": customer_id}))
        return orders
    except Exception as e:
        st.error(f"Error finding customer orders: {e}")
        return []

def update_order_status(order_id, new_status):
    """Update order status"""
    try:
        result = orders_collection.update_one(
            {"OrderID": order_id},
            {"$set": {"status": new_status, "updated_at": datetime.now().strftime("%Y-%m-%d")}}
        )
        return result.modified_count > 0
    except Exception as e:
        st.error(f"Error updating order: {e}")
        return False

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Load and prepare training data
@st.cache_data
def load_training_data():
    df = pd.read_csv('Bitext_Sample_Customer_Service_Testing_Dataset.csv')
    return df

# Train intent classifier
@st.cache_resource
def train_intent_classifier(df):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['utterance'])
    y = df['intent']
    model = LinearSVC(random_state=42)
    model.fit(X, y)
    return vectorizer, model

# Initialize training data and model
try:
    df = load_training_data()
    vectorizer, intent_classifier = train_intent_classifier(df)
except Exception as e:
    st.error(f"Error loading training data: {e}")
    st.stop()

# Configure Groq client
api_key = "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"  # Replace with your Groq API key
client = Groq(api_key=api_key)

# Initialize the model name
MODEL_NAME = "llama-3.3-70b-versatile"  # Current supported Groq model

# --- Intent Keywords ---
intent_keywords = {
    "order status": ["order", "track", "where is my order", "order status", "tracking"],
    "product inquiry": ["tell me about", "product", "features", "specifications"],
    "billing question": ["bill", "payment", "invoice", "charged"],
    "technical issue": ["error", "not working", "help", "problem"],
    "escalate to human": ["speak to human", "representative", "agent", "person"],
    "greeting": ["hi", "hello", "hey", "good morning"],
    "goodbye": ["bye", "goodbye", "thanks", "thank you"]
}

# --- Simple Intent Detection ---
def detect_intent(user_input):
    user_input = user_input.lower()
    for intent, keywords in intent_keywords.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", user_input):
                return intent
    return "escalate to human"

# --- Sentiment Analysis (Basic) ---
def analyze_sentiment(user_input):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(user_input)['compound']
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"

# Predict intent
def predict_intent(text):
    X = vectorizer.transform([text])
    intent = intent_classifier.predict(X)[0]
    return intent

# Get example responses for intent
def get_intent_examples(intent, n=3):
    examples = df[df['intent'] == intent]['utterance'].sample(n=min(n, len(df[df['intent'] == intent]))).tolist()
    return examples

# --- Main Response Logic ---
def generate_response(user_input):
    # Analyze sentiment and predict intent
    sentiment = analyze_sentiment(user_input)
    intent = predict_intent(user_input)
    examples = get_intent_examples(intent)
    
    # Display analytics
    # Also check against the keyword-based intent
    keyword_intent = detect_intent(user_input)
    if keyword_intent != "escalate to human":
        intent = keyword_intent  # Use the keyword-based intent if it's more specific
    
    # Extract order ID if present in the input - check for UUIDs or simple numeric order IDs
    order_id_match = re.search(r'order(?:\s+(?:id|number|#))?\s*[:#]?\s*([a-zA-Z0-9-]+)', user_input.lower())
    order_details = None
    
    # Also try to extract UUID-like patterns that might be order IDs
    if not order_id_match:
        uuid_pattern = r'\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b'
        uuid_match = re.search(uuid_pattern, user_input.lower())
        if uuid_match:
            order_id_match = uuid_match
    
    if order_id_match:
        order_id = order_id_match.group(1)
        # Look up the order in our database
        order_details = find_order(order_id)
        if order_details:
            st.write("Order Details Found:")
            st.json(order_details)
            
            # Display a formatted table of important order information
            order_status = order_details.get("status", "Unknown")
            delivery = order_details.get("delivery", "Unknown")
            customer_name = order_details.get("customer_details", {}).get("Name", "Unknown")
            
            st.markdown(f"""
            ### Order Summary
            | Order ID | Customer | Status | Delivery |
            | -------- | -------- | ------ | -------- |
            | {order_id} | {customer_name} | {order_status} | {delivery} |
            """)
        else:
            # If no order found, display a helpful message
            st.warning(f"Order ID {order_id} not found in our system.")
    
    try:
        # Create a detailed prompt with intent and examples
        prompt = f"""You are a helpful customer support agent. 
The customer's intent is: {intent}
The customer's sentiment is: {sentiment}
Similar customer queries in our database:
{' - ' + chr(10).join(examples)}

The customer says: '{user_input}'

{"Order details found: " + json.dumps(order_details) if order_details else "No specific order details found."}

Based on the intent, sentiment, and order details (if any), provide a helpful and relevant response. 
If the intent is a complaint or negative sentiment, show extra empathy."""

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent trained to handle various customer service scenarios."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response from Groq: {e}")
        return "I'm having trouble connecting to my brain right now. Please try again in a moment."

# --- Streamlit UI ---
st.title("Agentic AI Support Bot (Powered by Groq)")
st.write("👋 Hi! I'm your customer support assistant. How can I help you today?")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "order_id" not in st.session_state:
    st.session_state.order_id = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Get sentiment and intent for display
        sentiment = analyze_sentiment(prompt)
        intent = predict_intent(prompt)
        keyword_intent = detect_intent(prompt)
        
        # Use keyword-based intent if it's more specific
        if keyword_intent != "escalate to human":
            display_intent = keyword_intent
        else:
            display_intent = intent
            
        # Display analytics in a more user-friendly way
        st.info(f"📊 Analysis: Sentiment: **{sentiment}** | Detected Intent: **{display_intent}**")
        
        # Generate and display response
        response = generate_response(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Update order_id in session state if found
    intent = detect_intent(prompt)
    if intent == "order status":
        match = re.search(r"order (\d+)", prompt)
        if match:
            st.session_state.order_id = match.group(1)

# Add a sidebar with helpful information
with st.sidebar:
    st.header("Quick Tips")
    st.markdown("""
    You can ask me about:
    - Order status and tracking
    - Cancellations and refunds
    - Shipping information
    - Payment issues
    - General product questions
    
    Try asking things like:
    - "Hi, how are you?"
    - "Can you help me track my order?"
    - "I need to cancel my order"
    - "What's the status of order 48efa9e9-38d2-47a3-b549-96354d0a9792?"
    """)
    
    # Display sample order IDs for testing
    st.markdown("### Sample Order IDs for Testing")
    try:
        with open('sample_order_ids.txt', 'r') as f:
            sample_ids = f.readlines()
            for i, order_id in enumerate(sample_ids[:5]):  # Show first 5 only
                st.code(order_id.strip(), language="text")
    except Exception as e:
        st.warning(f"Could not load sample order IDs: {e}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
