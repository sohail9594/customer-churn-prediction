import streamlit as st
import time
import random
import sys
import os

# FIX: ADD PROJECT ROOT TO PATH FOR MODULE RESOLUTION 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.realtime import predict_realtime 


# Generate Random Incoming Data
def generate_random_customer():
    """Generates a random set of customer features for simulation."""
    return {
        "gender": random.choice(["Male", "Female"]),
        "paymentmethod": random.choice(["UPI", "Credit Card", "Credit card (automatic)", "Debit Card", "Net Banking", "PayPal", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Cash on Delivery" ]),
        "industry": random.choice(["Ecommerce", "Subscription", "Telecom"]),
        "age": random.randint(20, 70),
        "tenure": random.randint(1, 60),
        "monthlycharges": random.randint(20, 130) 
    }


# Helper function for input display
def display_detail(label, value, col):
    """Displays a label and value inside a styled Streamlit container column, mimicking a textbox look."""
    with col:
        st.markdown(f"**{label}:**")
        st.code(str(value), language='text')


# Streamlit UI
st.set_page_config(page_title="ChurnGuard Dashboard", layout="wide")

# 1. TITLE AND TAGLINE
st.title("ChurnGuard: Customer Churn Prediction")
st.markdown("### *Predict. Prevent. Retain.*")

# FIX: Replaced st.spacer() with st.markdown("") for cross-version compatibility (creates space)
st.markdown("", unsafe_allow_html=True)
st.markdown("", unsafe_allow_html=True)

placeholder = st.empty()

while True:
    with placeholder.container():
        # Generate simulated incoming data
        incoming = generate_random_customer()

        # ----------------------------------------------------------------------
        # 2. CUSTOMER DETAILS (INPUT) SECTION
        # ----------------------------------------------------------------------
        st.header("Customer Details")
        
        # Display details one per line using a two-column layout
        col_a, col_b = st.columns(2)

        # Left Column Details
        display_detail("Gender", incoming['gender'], col_a)
        display_detail("Payment Method", incoming['paymentmethod'], col_a)
        display_detail("Industry", incoming['industry'], col_a)
        
        # Right Column Details
        display_detail("Age (Years)", incoming['age'], col_b)
        display_detail("Tenure (Months)", incoming['tenure'], col_b)
        display_detail("Monthly Charges ($)", f"${incoming['monthlycharges']:.2f}", col_b)
        
        # Add a visual separator
        st.markdown("---") 

        # ----------------------------------------------------------------------
        # 3. MODEL RESULTS (OUTPUT) SECTION
        # ----------------------------------------------------------------------
        
        # Predict using backend
        output = predict_realtime(incoming)
        
        # Determine visual style based on prediction
        is_churn = output["prediction"] == 1
        churn_status = "Yes" if is_churn else "No"
        
        st.header("Results")
        
        # Using 4 columns for the key metrics for a clean, spaced-out look
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            # Reverted to default label size (removed HTML tags)
            st.metric(
                label="Churn Prediction", 
                value=churn_status,
                delta="HIGH RISK" if is_churn else "LOW RISK",
                delta_color="inverse" if is_churn else "normal"
            )

        with metric_col2:
            # Reverted to default label size (removed HTML tags)
            st.metric(
                label="Churn Probability", 
                value=f"{output['probability']:.2f}",
            )

        with metric_col3:
            # Customer Segment Metric (already plain text)
            st.metric("Customer Segment", value=output["segment"]) 

        with metric_col4:
            # Reverted to default label size (removed HTML tags)
            st.metric(
                label="CLV Estimate", 
                value=f"${output['clv']:.0f}"
            )

        # Recommended Action
        st.markdown("### Recommended Action:")
        
        action_text = output["recommended_action"]
        
        # Reverted to plain text inside the alert box (default size)
        if is_churn:
            st.warning(action_text)
        else:
            st.success(action_text)

        # Model Explanation
        st.markdown("### Model Explanation:") 
        
        # Reverted to plain text inside the alert box (default size)
        st.info(output["shap_explanation"])

    # Wait for 10 seconds before refreshing
    time.sleep(10)