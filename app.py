import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go


st.markdown("""
    <style>
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #151515;
        padding: 20px;
    }
    .sidebar .stSlider {
        color: red;
    }
    .sidebar .stMarkdown h2 {
        color: red;
        font-weight: bold;
    }

    /* Center header styling */
    .main .stMarkdown h1 {
        font-family: Arial, sans-serif;
        color: #fff;
        font-size: 20px;
        text-align: center;
        margin-bottom: 10px;
    }

    .main .stMarkdown p {
        font-family: Arial, sans-serif;
        color: #d9d9d9;
        font-size: 20px;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Main content styling */
    .main {
        background-color: #202020;
    }

    .stPlotlyChart {
        background-color: #333;
        padding: 10px;
        border-radius: 10px;
    }

    /* Prediction result styling */
    .result-box {
        background-color: #2b2b2b;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    
    .rainfall-result {
        font-size: 30px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .rainfall-result.red {
        color: red;
    }
    
    .rainfall-result.green {
        color: green;
    }
    </style>
""", unsafe_allow_html=True)

# Function to create a radar chart using Plotly
def create_radar_chart(categories, values, title):
    """Create a radar chart using Plotly."""
    values += values[:1]  # Close the loop for radar chart
    categories += categories[:1]  # Close the loop for radar chart

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Input Features'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title=title
    )
    
    return fig

# Loading the model
def load_model(filepath):
    try:
        model_data = joblib.load("rainfall_prediction_model.pkl")
        return model_data["model"], model_data["features_names"]
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Title and Description
st.markdown("<h1>Rainfall Prediction System</h1>", unsafe_allow_html=True)
st.markdown("""
    <p>Please connect this app to your environmental data to predict whether rainfall is likely. Adjust the parameters in the sidebar to update the inputs and view the prediction results.</p>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("Environmental Measurements")

# Load the model and feature names
model, feature_names = load_model("rainfall_prediction_model.pkl")

# Input sliders for the features
input_values = []
for feature in feature_names:
    if feature == "pressure":
        value = st.sidebar.slider("Pressure (hPa)", 950, 1050, 1010)
    elif feature == "dewpoint":
        value = st.sidebar.slider("Dew Point (°C)", -10, 50, 20)
    elif feature == "humidity":
        value = st.sidebar.slider("Humidity (%)", 0, 100, 50)
    elif feature == "cloud":
        value = st.sidebar.slider("Cloud Cover (%)", 0, 100, 50)
    elif feature == "sunshine":
        value = st.sidebar.slider("Sunshine (hours)", 0.0, 24.0, 5.0)
    elif feature == "winddirection":
        value = st.sidebar.slider("Wind Direction (degrees)", 0, 360, 180)
    elif feature == "windspeed":
        value = st.sidebar.slider("Wind Speed (m/s)", 0.0, 25.0, 5.0)
    else:
        value = st.sidebar.slider(feature, 0, 100, 50)
    input_values.append(value)

# Create a radar chart for input features
st.subheader("Input Feature Visualization")
fig = create_radar_chart(feature_names, input_values, "Input Feature Radar Chart")
st.plotly_chart(fig)

# Prediction button and result display
if st.sidebar.button("Predict"):
    try:
        # Prepare input data for prediction
        input_data = np.array(input_values[:7]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)

        # Display prediction result
        st.subheader("Prediction Results")
        result_color = "red" if prediction[0] == 1 else "green"
        result_text = "Rainfall is Likely" if prediction[0] == 1 else "No Rainfall Expected"
        
        st.markdown(f"""
            <div class="result-box">
                <p class="rainfall-result {result_color}">{result_text}</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

