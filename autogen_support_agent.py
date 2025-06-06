import streamlit as st
import autogen
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from groq import Groq
import os

# Download VADER lexicon if not already present
nltk.download('vader_lexicon', quiet=True)

# Groq configuration
groq_config = {
    "api_key": "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",
    "model": "llama-3.3-70b-versatile"
}

# Configure Groq client for the agents
groq_client = Groq(api_key=groq_config["api_key"])

# Configure the LLM config for agents
llm_config = {
    "config_list": [{
        "model": groq_config["model"],
        "api_key": groq_config["api_key"]
    }]
}

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

# Create the agent system
def create_agent_system():
    # Customer Service Agent
    customer_service_agent = autogen.AssistantAgent(
        name="customer_service",
        system_message="You are a helpful and professional customer service agent. \
                       You provide clear, concise, and helpful responses to customer inquiries.",
        llm_config=llm_config
    )

    # Sentiment Analyzer Agent
    sentiment_agent = autogen.AssistantAgent(
        name="sentiment_analyzer",
        system_message="You analyze customer sentiment and provide guidance on tone and approach.",
        llm_config=llm_config
    )

    # User Proxy Agent
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda x: True  # Terminate after one response
    )

    return customer_service_agent, sentiment_agent, user_proxy

# Generate response using the agent system
def generate_response(user_input):
    # Create the agent system
    customer_service_agent, sentiment_agent, user_proxy = create_agent_system()
    
    # Analyze sentiment
    sentiment = analyze_sentiment(user_input)
    st.write(f"Detected sentiment: {sentiment}")
    
    try:
        # Prepare the message with sentiment context
        message = f"""
        Customer message: '{user_input}'
        Detected sentiment: {sentiment}
        
        Please provide a helpful and appropriate response considering the customer's sentiment.
        """
        
        # Initiate the chat
        user_proxy.initiate_chat(
            customer_service_agent,
            message=message
        )
        
        # Get the last message from the conversation (the agent's response)
        chat_history = user_proxy.chat_messages[customer_service_agent]
        response = chat_history[-1]["content"]
        
        return response
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm having trouble connecting to my brain right now. Please try again in a moment."

# --- Streamlit UI ---
def main():
    st.title("Advanced AI Support Bot (Powered by Autogen & Groq)")
    st.write("Ask me anything related to your order, product, billing, or account.")
    st.write("This version uses multiple AI agents working together to provide better support!")

    user_input = st.text_input("You:")
    if user_input:
        response = generate_response(user_input)
        st.text_area("Bot:", value=response, height=150, max_chars=None)

if __name__ == "__main__":
    main()
