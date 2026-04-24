import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import time

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Power Forecaster", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a slight UI tweak (optional but makes buttons look better)
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_features = joblib.load('selected_features.pkl')
    return model, scaler, selected_features

model, scaler, selected_features = load_assets()

original_features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
                     'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 
                     'hour', 'day_of_week', 'lag_1', 'lag_24', 'rolling_mean_24']

# ---------------------------------------------------------
# Sidebar Layout
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=80)
    st.title("Model Dashboard")
    st.info("This application utilizes a Ridge Regression model to predict short-term household electricity consumption based on historical time-series data.")
    
    st.markdown("### 🧠 Model Architecture")
    st.markdown("- **Algorithm:** Ridge Regression\n- **Dimensionality:** Forward Feature Selection\n- **Cross-Validation:** TimeSeriesSplit")
    
    st.markdown("---")
    st.caption("Developed With ❤️ by Sahil Singh")

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.title("⚡ Next-Hour Electricity Usage Forecaster")
st.markdown("Enter the current environmental and power readings below to generate a real-time prediction for the next hour's grid load.")
st.divider()

# Organize inputs into clean, logical sections
st.subheader("📊 Current Environment Inputs")

with st.container():
    st.markdown("**1. Grid Metrics**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        global_active_power = st.number_input("Active Power (kW)", value=1.50, help="Total active power consumed by the household")
    with col2:
        global_reactive_power = st.number_input("Reactive Power (kW)", value=0.10, help="Total reactive power (wasted power)")
    with col3:
        voltage = st.number_input("Voltage (V)", value=240.0, help="Average grid voltage")
    with col4:
        global_intensity = st.number_input("Intensity (A)", value=6.0, help="Current intensity")

st.write("") # Spacer

with st.container():
    st.markdown("**2. Zonal Sub-Metering (Wh)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        sub_metering_1 = st.number_input("Kitchen & Laundry (Sub 1)", value=0.0)
    with col2:
        sub_metering_2 = st.number_input("Living Room (Sub 2)", value=1.0)
    with col3:
        sub_metering_3 = st.number_input("Water Heater/AC (Sub 3)", value=18.0)

st.write("") # Spacer

with st.container():
    st.markdown("**3. Temporal & Historical Trends**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        hour = st.number_input("Current Hour", min_value=0, max_value=23, value=datetime.now().hour)
    with col2:
        day = st.number_input("Day (0=Mon)", min_value=0, max_value=6, value=datetime.now().weekday())
    with col3:
        lag_1 = st.number_input("Prev. Hour Load", value=1.40)
    with col4:
        lag_24 = st.number_input("Load 24h Ago", value=1.60)
    with col5:
        rolling_mean_24 = st.number_input("24h Average", value=1.30)

st.divider()

# ---------------------------------------------------------
# Prediction Execution
# ---------------------------------------------------------
# Make the button span the whole container
if st.button("🔮 Generate Prediction", use_container_width=True):
    
    # Add a spinner to simulate processing time for a professional feel
    with st.spinner('Standardizing inputs and running model inference...'):
        time.sleep(1) # Brief pause for UX
        
        # Compile inputs into a dataframe
        input_data = pd.DataFrame([[
            global_active_power, global_reactive_power, voltage, global_intensity,
            sub_metering_1, sub_metering_2, sub_metering_3,
            hour, day, lag_1, lag_24, rolling_mean_24
        ]], columns=original_features)
        
        # Scale inputs
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=original_features)
        
        # Filter to selected features only
        final_input = input_scaled[selected_features]
        
        # Predict
        prediction = model.predict(final_input)[0]
        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]
            
        # Ensure prediction doesn't go below zero
        final_prediction = max(0, prediction)

    # Display result using a nice dashboard metric
    st.success("Inference Complete!")
    st.metric(label="Predicted Next-Hour Consumption", value=f"{final_prediction:.3f} kW", delta="Estimated Load")
    
    # Add an expander for transparency (examiners love this)
    with st.expander("View Standardized Model Inputs"):
        st.dataframe(final_input.style.format("{:.4f}"))