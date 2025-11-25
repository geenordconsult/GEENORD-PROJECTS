import streamlit as st
import joblib
import pandas as pd
import numpy as np
 

# --- 1. CONFIGURATION AND MODEL LOADING ---

# Define the features your model was trained on, IN THE EXACT ORDER
FEATURE_ORDER = ['N', 'P', 'K', 'ph'] 

# Dictionary for clean UI labels and help text
DISPLAY_NAMES = {
    'N': 'Nitrogen (N) Content',
    'P': 'Phosphorus (P) Content',
    'K': 'Potassium (K) Content',
    'ph': 'Soil pH Level'
}

# Define agricultural acceptable ranges for validation (Based on search results)
INPUT_RANGES = {
    'N': {'min': 0.0, 'max': 200.0, 'default': 90.0, 'help': 'Available Nitrogen in soil (e.g., mg/kg or ppm).'},
    'P': {'min': 0.0, 'max': 200.0, 'default': 42.0, 'help': 'Available Phosphorus in soil (e.g., mg/kg or ppm).'},
    'K': {'min': 0.0, 'max': 300.0, 'default': 43.0, 'help': 'Available Potassium in soil (e.g., mg/kg or ppm).'},
    'ph': {'min': 3.5, 'max': 10.0, 'default': 5.5, 'help': 'Soil acidity/alkalinity level (3.5 = highly acidic, 9.5 = highly alkaline).'}
}

# --- CROP MAPPING (Crucial Fix) ---
# This dictionary maps the numeric labels predicted by your model to the actual crop names.
CROP_MAPPING = {
    20: 'Rice',
    11: 'Maize',
    3: 'Chickpea',
    9: 'Kidneybeans',
    18: 'Pigeonpeas',
    13: 'Mothbeans',
    14: 'Mungbean',
    2: 'Blackgram',
    10: 'Lentil',
    19: 'Pomegranate',
    1: 'Banana',
    12: 'Mango',
    7: 'Grapes',
    21: 'Watermelon',
    15: 'Muskmelon',
    0: 'Apple',
    16: 'Orange',
    17: 'Papaya',
    4: 'Coconut',
    6: 'Cotton',
    8: 'Jute',
    5: 'Coffee'
}

@st.cache_resource
def load_model():
    """Load the model file and optional scaler/encoder only once."""
    try:
        model = joblib.load('best_model.pkl')
        return model
    except FileNotFoundError:
        # st.stop() halts the script execution immediately, which is ideal here
        st.error("Error: Model file 'best_model.pkl' not found. Please ensure it is in the same directory as app.py and committed to your repository.")
        st.stop()
        
model = load_model()

# --- 2. STREAMLIT UI SETUP ---

st.set_page_config(page_title="Crop Recommendation App by Engr. Michael Jinadu", layout="centered", initial_sidebar_state="collapsed")
st.title("🌾 Soil Fertility Crop Recommender")

# Sidebar for context and branding
st.sidebar.header("About This Tool")
st.sidebar.info(
   "This application uses a trained **Machine Learning Classification Model** "
    "to recommend the most suitable crop based on the N-P-K-pH soil test results. "
    
    "\n\n---"
    "\n**Model Training & Performance**"
    "\n\n- The model was trained on a dataset gotten from Kaggle/DataCamp."
    f"\n- The Accuracy of this Predictive Model is **80%**."
    "\n- This Model was trained to predict the most suitable crop from the following 22 types: "
    
    "\n\n\t**PULSES/GRAINS:** rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil"
    "\n\n\t**FRUITS:** pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut"
    "\n\n\t**FIBRE/OTHER:** cotton, jute, coffee"
)
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Geenord Tech Solutions")


# Input Form
with st.form("input_form"):
    st.header("🔬 Soil Nutrient Parameters")
    
    # Organize inputs using two columns
    col1, col2 = st.columns(2)
    
    with col1:
        N = st.number_input(f"1. {DISPLAY_NAMES['N']}", 
                            min_value=INPUT_RANGES['N']['min'], 
                            max_value=INPUT_RANGES['N']['max'], 
                            value=INPUT_RANGES['N']['default'],
                            help=INPUT_RANGES['N']['help'])
        
        K = st.number_input(f"3. {DISPLAY_NAMES['K']}", 
                            min_value=INPUT_RANGES['K']['min'], 
                            max_value=INPUT_RANGES['K']['max'], 
                            value=INPUT_RANGES['K']['default'], 
                            help=INPUT_RANGES['K']['help'])
        
    with col2:
        P = st.number_input(f"2. {DISPLAY_NAMES['P']}", 
                            min_value=INPUT_RANGES['P']['min'], 
                            max_value=INPUT_RANGES['P']['max'], 
                            value=INPUT_RANGES['P']['default'], 
                            help=INPUT_RANGES['P']['help'])
        
        ph = st.number_input(f"4. {DISPLAY_NAMES['ph']}", 
                             min_value=INPUT_RANGES['ph']['min'], 
                             max_value=INPUT_RANGES['ph']['max'], 
                             value=INPUT_RANGES['ph']['default'], 
                             help=INPUT_RANGES['ph']['help'],
                             format="%.2f") # Ensure pH shows two decimal places

    submitted = st.form_submit_button("Get Crop Recommendation 🚀")

# --- 3. PREDICTION LOGIC AND UX ---

if submitted:
    # 1. Gather all inputs into a dictionary
    input_features = {
        'N': N, 
        'P': P, 
        'K': K, 
        'ph': ph
    }

    # 2. Input Validation (Checking for extreme/invalid values)
    is_valid = True
    if ph < 4.0 or ph > 8.5:
        st.error("⚠️ Warning: pH value is outside the optimal range (4.0 - 8.5) for most major crops. Prediction may be inaccurate.")
        is_valid = False

    if N == 0 or P == 0 or K == 0:
        st.warning("🚨 Caution: Zero values for major nutrients (N, P, K) might indicate an erroneous test result or severely depleted soil. Proceeding with prediction.")
        
    # Proceed only if the core validation passes (we continue with prediction for warnings)
    
    if is_valid:
        with st.spinner('Calculating optimal crop recommendation...'):
            
            # Create a Pandas DataFrame
            input_df = pd.DataFrame([input_features], columns=FEATURE_ORDER)
            
            # ** IMPORTANT: Apply Scaler/Encoder here if used during training **
            # If you scaled your features, load your scaler:
            # scaler = joblib.load('scaler.pkl') 
            # input_df = scaler.transform(input_df)

            # Make the prediction
            try:
                # 1. Get the numeric prediction (e.g., 20, 11, 3, etc.)
                numeric_prediction = model.predict(input_df)[0]
                
                # Convert the model's output to an integer to match the dictionary keys
                numeric_prediction = int(numeric_prediction) 

                # 2. Look up the crop name using the official mapping
                predicted_crop_name = CROP_MAPPING.get(numeric_prediction, "Unknown Crop (Error)")
                
                # 3. Display the result
                if predicted_crop_name != "Unknown Crop (Error)":
                    st.success(f"### 🎉 Optimal Crop Recommendation: **{predicted_crop_name.title()}**")
                    st.balloons()
                else:
                    st.error(f"Error: Model predicted an unknown label index: {numeric_prediction}. Check your CROP_MAPPING.")
                
                st.info(f"Model internal output (Label Index): {numeric_prediction}")
                
            except Exception as e:
                st.error("Prediction failed. Check your model input order and data types.")
                st.exception(e)

        # 4. Display Input Summary for Verification (UX Enhancement)
        with st.expander("Show Input Details"):
            # Transpose DataFrame for cleaner vertical display
            display_df = input_df.T
            display_df = display_df.rename(columns={display_df.columns[0]: 'Input Value'})
            display_df.index = [DISPLAY_NAMES[col] for col in display_df.index]
            
            st.dataframe(display_df, use_container_width=True)
            st.caption("Values used to generate the recommendation.")

# --- 4. DISCLAIMER ---
st.markdown("---")
st.caption(
    "Disclaimer: This recommendation is generated by a Machine Learning model and should be "
    "used for guidance only. Always consult with a local agricultural extension officer "
    "and conduct full soil analysis before making final planting decisions."
)