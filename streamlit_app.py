import streamlit as st

st.set_page_config(
    page_title="ChurnGuard Main App",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation([
    st.Page("pages/Real_Time_Dashboard.py", title="Real-Time Dashboard", icon="📡"),
    st.Page("pages/Manual_Prediction.py",   title="Manual Prediction",   icon="🔍"),
])
pg.run()

'''
st.title("ChurnGuard: Customer Churn Prediction")
st.markdown("### Welcome")
st.markdown("Use the navigation links in the sidebar to switch between the following tabs:")
st.markdown("""
* **Manual Prediction:** Allows you to input custom customer features and receive an immediate prediction.
* **Real-Time Dashboard:** Simulates a live data stream to show continuous predictions.
""")
'''