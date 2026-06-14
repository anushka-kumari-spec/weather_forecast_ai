import streamlit as st
import pandas as pd
import joblib

# Load model and label encoder
model = joblib.load("model/weather_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Optimally cache data loading to prevent the app from slowing down
@st.cache_data
def load_training_data():
    try:
        return pd.read_csv("data/weather_data.csv")
    except FileNotFoundError:
        return None

df = load_training_data()

# Page Config (Set to 'wide' to accommodate charts cleanly)
st.set_page_config(
    page_title="Weather Prediction AI",
    page_icon="🌦️",
    layout="wide"
)

# Load local CSS styles (if present)
def load_css(file_path: str = "styles.css"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# Theme toggle in sidebar
with st.sidebar:
    st.markdown("## Display")
    dark_mode = st.checkbox("Enable Dark Mode", value=False)

load_css("styles_dark.css" if 'dark_mode' in globals() and dark_mode else "styles.css")

st.title("🌦️ Weather Prediction AI & Analytics Dashboard")
st.write("Predict weather conditions or explore trends inside the training dataset.")

# Setup Layout Tabs to split Prediction UI from Data Analysis
tab1, tab2 = st.tabs(["🔮 Predict Weather", "📊 Data Analytics"])

# =====================================================================
# TAB 1: PREDICTION INTERFACE
# =====================================================================
with tab1:
    st.header("Predict Weather Condition")
    st.write("Enter specific environmental values to evaluate current weather trends:")
    
    # Quick preset buttons
    presets = st.columns(3)
    if presets[0].button("Set Cold"):
        st.session_state.temperature = 5.0
        st.session_state.min_temp = 0.0
        st.session_state.max_temp = 10.0
        st.session_state.avg_temp = 5.0
    if presets[1].button("Set Warm"):
        st.session_state.temperature = 28.0
        st.session_state.min_temp = 20.0
        st.session_state.max_temp = 32.0
        st.session_state.avg_temp = 27.0
    if presets[2].button("Set Hot"):
        st.session_state.temperature = 40.0
        st.session_state.min_temp = 35.0
        st.session_state.max_temp = 45.0
        st.session_state.avg_temp = 40.0

    # Split input fields into columns for a cleaner layout and persist values in session state
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Current Temperature (°C)", min_value=-10.0, max_value=60.0, value=30.0, key="temperature")
        st.number_input("Minimum Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, key="min_temp")
    with col2:
        st.number_input("Maximum Temperature (°C)", min_value=-10.0, max_value=60.0, value=35.0, key="max_temp")
        st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=60.0, value=30.0, key="avg_temp")

    # Local copies for prediction logic
    temperature = st.session_state.get("temperature", 30.0)
    min_temp = st.session_state.get("min_temp", 25.0)
    max_temp = st.session_state.get("max_temp", 35.0)
    avg_temp = st.session_state.get("avg_temp", 30.0)

    st.markdown("---")

    # Prediction Button
    if st.button("Predict Weather", type="primary"):
        input_data = pd.DataFrame({
            "Temperature_C": [temperature],
            "Min_Temp_C": [min_temp],
            "Max_Temp_C": [max_temp],
            "Avg_Temp_C": [avg_temp]
        })

        prediction = model.predict(input_data)
        weather_label = label_encoder.inverse_transform(prediction)[0]

        st.success(f"**Predicted Weather:** {weather_label}")

        if weather_label == "Cold":
            st.info("🥶 Weather is Cold")
        elif weather_label == "Normal":
            st.info("😊 Weather is Normal")
        elif weather_label == "Warm":
            st.warning("🌤️ Weather is Warm")
        elif weather_label == "Hot":
            st.error("🔥 Weather is Hot")


# =====================================================================
# TAB 2: DATA ANALYTICS & GRAPH SECTION
# =====================================================================
with tab2:
    st.header("Historical Training Data Analytics")
    
    if df is not None:
        # 1. Summary Metrics Cards
        st.subheader("Key Dataset Benchmarks")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        metric_col1.metric("Total Records", f"{len(df)}")
        metric_col2.metric("Dataset Avg Temp", f"{df['Avg_Temp_C'].mean():.1f}°C")
        metric_col3.metric("Highest Max Temp", f"{df['Max_Temp_C'].max():.1f}°C")
        metric_col4.metric("Lowest Min Temp", f"{df['Min_Temp_C'].min():.1f}°C")
        
        st.markdown("---")
        
        # 2. Analytics Trend Line Chart
        st.subheader("📈 Temperature Trend Profile")
        st.write("An analytics line graph tracking the spread between Min, Max, and Avg temperatures across the historical timeline.")
        # Passing relevant columns to construct a combined interactive line chart
        st.line_chart(df[['Min_Temp_C', 'Avg_Temp_C', 'Max_Temp_C']])
        
        # Extra area chart for overall average temperatures
        st.markdown("---")
        st.subheader("🌡️ Average Temperature Over Time")
        st.area_chart(df['Avg_Temp_C'])
        
        st.markdown("---")
        st.markdown("---")
        
        # 3. Bar Chart Comparison
        st.subheader("📊 Temperature Distribution Breakdown")
        st.write("A bar graph mapping statistical thresholds parsed directly from your dataset.")
        
        summary_stats = pd.DataFrame({
            'Thresholds': ['Record Low', 'Average Minimum', 'Overall Average', 'Average Maximum', 'Record High'],
            'Temperature (°C)': [
                df['Min_Temp_C'].min(), 
                df['Min_Temp_C'].mean(), 
                df['Avg_Temp_C'].mean(), 
                df['Max_Temp_C'].mean(), 
                df['Max_Temp_C'].max()
            ]
        }).set_index('Thresholds')
        
        st.bar_chart(summary_stats)
        
        # 4. Optional Raw Data Viewer Toggle
        st.markdown("---")
        if st.checkbox("Show Raw Training Data Snippet"):
            st.subheader("Training Data Preview")
            st.dataframe(df.head(25))
            
    else:
        st.error("Unable to extract historical data. Verify that `weather_data.csv` is correctly positioned inside the `data/` folder directory.")