import streamlit as st
import pickle
import time

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System - Team Tigers",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title-text {
        text-align: center;
        font-size: clamp(2rem, 4vw, 3rem);
        color: #2e7d32;
        margin-bottom: 1rem;
    }
    .subtitle-text {
        text-align: center;
        font-size: clamp(1rem, 2vw, 1.5rem);
        color: #666;
        margin-bottom: 1rem;
    }
    .success-text {
        color: #4CAF50;
        font-size: clamp(1rem, 2vw, 1.2rem);
        font-weight: bold;
    }
    .warning {
        color: #ff4444;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    div[data-baseweb="input"] input {
        background-color: transparent !important;
    }
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        .stButton>button {
            margin-top: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model"""
    try:
        with open('lgb_model.pickle', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_crop(model, features):
    """Make prediction using the model"""
    try:
        predicted_crop = model.predict([features])
        return predicted_crop.item()
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def validate_inputs(inputs):
    """Validate that all inputs have been provided"""
    validation_messages = []
    input_names = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", 
                  "Humidity", "pH Level", "Rainfall"]
    
    for value, name in zip(inputs, input_names):
        if value is None or value == "":
            validation_messages.append(f"{name} is required")
    
    return validation_messages

def main():
    # Header
    st.markdown("<h1 class='title-text'>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Powered by Machine Learning with 99% Accuracy</p>", unsafe_allow_html=True)

    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if 'model.pickle' exists.")
        return

    # Create responsive columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enter Soil and Climate Parameters")
        
        # Input fields with empty default values
        n = st.text_input("Nitrogen (N) - mg/kg", 
                       help="Enter the nitrogen content in mg/kg",
                       placeholder="Enter value between 0-140")
        
        p = st.text_input("Phosphorus (P) - mg/kg", 
                       help="Enter the phosphorus content in mg/kg",
                       placeholder="Enter value between 0-140")
        
        k = st.text_input("Potassium (K) - mg/kg", 
                       help="Enter the potassium content in mg/kg",
                       placeholder="Enter value between 0-200")
        
        temperature = st.text_input("Temperature (°C)", 
                                help="Enter the temperature in Celsius",
                                placeholder="Enter value between 0-50")
        
        humidity = st.text_input("Humidity (%)", 
                             help="Enter the humidity percentage",
                             placeholder="Enter value between 0-100")
        
        ph = st.text_input("pH Level", 
                        help="Enter the pH level of soil",
                        placeholder="Enter value between 0-14")
        
        rainfall = st.text_input("Rainfall (mm)", 
                             help="Enter the rainfall in millimeters",
                             placeholder="Enter value between 0-300")

    with col2:
        st.subheader("About")
        st.write("""
        This Crop Recommendation System uses advanced machine learning techniques 
        to suggest the most suitable crop based on soil parameters and climate conditions.
        
        Our model achieves 99% accuracy in predicting the optimal crop for your agricultural needs.
        """)
        
        st.markdown("### Available Crops")
        crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
                'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
        st.write(", ".join(crops))

    # Predict button and validation
    if st.button("Predict Recommended Crop"):
        # Convert inputs to float and validate
        try:
            features = [float(x) if x else None for x in [n, p, k, temperature, humidity, ph, rainfall]]
            validation_messages = validate_inputs(features)
            
            if validation_messages:
                st.error("Please fix the following issues:")
                for msg in validation_messages:
                    st.markdown(f"<p class='warning'>• {msg}</p>", unsafe_allow_html=True)
            else:
                # Validate ranges
                ranges = [
                    (0, 140, "Nitrogen"), (0, 140, "Phosphorus"), 
                    (0, 200, "Potassium"), (0, 50, "Temperature"),
                    (0, 100, "Humidity"), (0, 14, "pH"),
                    (0, 300, "Rainfall")
                ]
                
                range_errors = []
                for value, (min_val, max_val, name) in zip(features, ranges):
                    if value < min_val or value > max_val:
                        range_errors.append(f"{name} must be between {min_val} and {max_val}")
                
                if range_errors:
                    st.error("Please fix the following issues:")
                    for msg in range_errors:
                        st.markdown(f"<p class='warning'>• {msg}</p>", unsafe_allow_html=True)
                else:
                    with st.spinner('Analyzing soil and climate parameters...'):
                        time.sleep(1)
                        prediction = predict_crop(model, features)
                        
                        if prediction:
                            st.success(f"### Recommended Crop: **{prediction.title()}** 🌱")
                            st.markdown(f"""
                            This recommendation is based on:
                            - Nitrogen: {n} mg/kg
                            - Phosphorus: {p} mg/kg
                            - Potassium: {k} mg/kg
                            - Temperature: {temperature}°C
                            - Humidity: {humidity}%
                            - pH Level: {ph}
                            - Rainfall: {rainfall} mm
                            """)
        
        except ValueError:
            st.error("Please ensure all inputs are valid numbers")

if __name__ == "__main__":
    main()