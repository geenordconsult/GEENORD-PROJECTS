import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
# Set the page configuration first
st.set_page_config(
    page_title="Customer Segmentation App | Geenord Tech Solutions", 
    layout="wide", # Use 'wide' layout for a better dashboard feel
    initial_sidebar_state="collapsed"
)

st.title("🛍️ Customer Segment Predictor")
st.markdown("---") # Use a separator for clean look

# --- Load Models (Assuming files exist) ---
try:
    # LOAD THE MODEL 
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("kscaler.pkl")
except FileNotFoundError:
    st.error("Model or Scaler files not found! Please ensure 'kmeans_model.pkl' and 'kscaler.pkl' are in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --- Custom Styling for Success/Error Messages ---
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        padding-top: 1rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.2rem;
        height: 3em;
        background-color: #4CAF50; /* Green */
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

st.header("CUSTOMER PROFILE INPUT")
st.write("Adjust the inputs below to define the customer's profile.")

# --- DATA INPUTS using Containers and Columns ---

# 1. Financial/Demographic Section
st.subheader("Financial & Demographic Data")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Customer's age.")
with col2:
    income = st.number_input("Income ($)", min_value=0, max_value=1000000, value=50000, help="Annual income of the customer.")
with col3:
    total_spending = st.number_input("Total Spending ($)", min_value=0, max_value=10000, value=2000, help="Total money spent across all product categories.")

st.markdown("---")

# 2. Activity/Engagement Section
st.subheader("Purchase & Activity Data")
col4, col5, col6, col7 = st.columns(4)

with col4:
    num_web_purchases = st.number_input("Web Purchases", min_value=0, max_value=100, value=10, help="Number of purchases made via the company website.")
with col5:
    num_store_purchases = st.number_input("Store Purchases", min_value=0, max_value=100, value=10, help="Number of purchases made in physical stores.")
with col6:
    num_web_visits = st.number_input("Web Visits (Monthly)", min_value=0, max_value=50, value=5, help="Number of website visits per month.")
with col7:
    recency = st.number_input("Recency (Days)", min_value=0, max_value=365, value=30, help="Days since the customer's last purchase. Lower is better (more recent).")

# --- Prepare Data ---
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spendings": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# --- Prediction and Output ---
st.markdown("---")
st.header("PREDICTION RESULT")

if st.button("PREDICT CUSTOMER SEGMENT"):
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0] # Get the single result

    # --- Result Display ---
    st.markdown("### SEGMENT PREDICTED:")
    
    # Use st.container to display the result prominently
    with st.container():
        st.markdown(f"## 🎯 Cluster **{cluster}**")

        # Define the segment descriptions and colors
        segment_info = {
            0: ("The Engaged Bargain Hunters", "Low Value, Active Buyers. Focus on Upselling."),
            1: ("At-Risk High Spenders", "High Value, Inactive Buyers. Immediate Win-Back Target."),
            2: ("Dormant/Lowest-Value Churn", "Low Value, Inactive Buyers. Minimize Marketing Spend."),
            3: ("Affluent Loyalists", "Highest Value, Active Buyers. Maximize Retention and Exclusive Offers.")
        }
        
        name, action = segment_info.get(cluster, ("Unknown Cluster", "No information available."))

        st.markdown(f"**Segment Name:** **{name}**")
        st.markdown(f"**Strategic Action:** {action}")
        
        # Optional: Show the input data used for transparency
        with st.expander("Show Input Data Used"):
            st.dataframe(input_data)


# --- Footer ---
st.markdown("---")
st.caption("Developed by Engr. Michael Jinadu | © 2025 Geenord Tech Solutions")