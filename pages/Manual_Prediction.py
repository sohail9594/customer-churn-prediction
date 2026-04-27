import streamlit as st
import sys
import os

# --- FIX: ADD PROJECT ROOT TO PATH FOR MODULE RESOLUTION ---
# Get the directory of the current script (pages/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.join(current_dir, '..')
# Add the project root to sys.path so 'backend' can be imported
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------------------------

from backend.realtime import predict_realtime 

# ---------------------------
# Helper function for result display
# ---------------------------
def display_results(output):
    """Displays the model output using the styled metrics and boxes."""
    st.markdown("---")
    st.header("Prediction Results")

    # Determine visual style based on prediction
    is_churn = output["prediction"] == 1
    churn_status = "Yes" if is_churn else "No"

    # Using 4 columns for the key metrics for a clean, spaced-out look
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Churn Prediction", 
            value=churn_status,
            delta="HIGH RISK" if is_churn else "LOW RISK",
            delta_color="inverse" if is_churn else "normal"
        )

    with metric_col2:
        st.metric(
            label="Churn Probability", 
            value=f"{output['probability']:.2f}",
        )

    with metric_col3:
        st.metric("Customer Segment", value=output["segment"]) 

    with metric_col4:
        st.metric(
            label="CLV Estimate", 
            value=f"${output['clv']:.0f}"
        )

    # Recommended Action
    st.markdown("### Recommended Action:")
    
    action_text = output["recommended_action"]
    
    if is_churn:
        st.warning(action_text)
    else:
        st.success(action_text)

    # Model Explanation
    st.markdown("### Model Explanation:") 
    st.info(output["shap_explanation"])


# ---------------------------
# Streamlit UI for Manual Prediction
# ---------------------------
st.title("ChurnGuard: Manual Churn Prediction")
st.markdown("### *Predict. Prevent. Retain.*")
st.markdown("Enter the customer details below and click 'Predict' to get an instant churn forecast.")

# Create the form for input
with st.form(key='churn_input_form'):
    st.subheader("Customer Features")
    col1, col2 = st.columns(2)

    # Left Column Inputs
    gender = col1.radio("Gender", ["Male", "Female"])
    industry = col1.selectbox("Industry", ["Ecommerce", "Subscription", "Telecom"])
    age = col1.slider("Age (Years)", 18, 99, 45)

    # Right Column Inputs
    paymentmethod = col2.selectbox("Payment Method", [
        "UPI", "Credit Card", "Credit card (automatic)", "Debit Card", "Net Banking", 
        "PayPal", "Electronic check", "Mailed check", "Bank transfer (automatic)", 
        "Cash on Delivery"
    ])
    tenure = col2.slider("Tenure (Months)", 0, 72, 24)
    monthlycharges = col2.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=65.0, step=1.0)

    # Submit Button
    submitted = st.form_submit_button("Run Prediction")

# Handle form submission
if submitted:
    input_data = {
        "gender": gender,
        "paymentmethod": paymentmethod,
        "industry": industry,
        "age": age,
        "tenure": tenure,
        "monthlycharges": monthlycharges
    }

    try:
        with st.spinner('Calculating Churn Prediction...'):
            # Call the backend function
            prediction_output = predict_realtime(input_data)
        
        display_results(prediction_output)
        
    except Exception as e:
        st.error(f"An error occurred during prediction. Please check the backend connection: {e}")