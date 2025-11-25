# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Tool for loading pre-trained models/scalers
from catboost import CatBoostClassifier


# --- HARDCODED PERFORMANCE METRICS ---
# These scores will be displayed beneath the Feature Importance chart.
ACCURACY = 0.9512
AUC_SCORE = 0.9460
PRECISION = 0.9338
RECALL = 0.9276
F1_SCORE = 0.9307
# -------------------------------------

# -------------------------------
# Load CatBoost model and Scaler
# -------------------------------

@st.cache_data
def load_assets():
    # 1. Load CatBoost model
    model = CatBoostClassifier(verbose=0)
    try:
        model.load_model("alzheimer_model.cbm")
    except Exception as e:
        st.error(f"Error loading CatBoost model: {e}")
        st.stop()
        
    # 2. Load StandardScaler (assuming you saved it with joblib/pickle)
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"Error loading StandardScaler: {e}")
        st.warning("Please ensure your StandardScaler object is saved as 'scaler.joblib' and available.")
        st.stop()
        
    return model, scaler

model, scaler = load_assets()


# ----------------------------------------------------
# Define Feature Groups for Data Transformation
# ----------------------------------------------------

# Features that were scaled (must be transformed by the scaler)
SCALED_COLS = [
    'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 
    'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL'
]

# Features that were left as integers (categorical/binary)
INT_CAT_COLS = [
    'EducationLevel', 'Gender', 'Ethnicity', 'Smoking', 'FamilyHistoryAlzheimers', 
    'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 
    'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation', 
    'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
]

# All features in the EXACT order the model expects (15 Scaled + 17 Categorical = 32)
# NOTE: The order established in the previous fix is preserved here.
EXPECTED_COLUMNS_ORDER = SCALED_COLS + INT_CAT_COLS

# -------------------------------
# App title and Inputs (Same as before)
# -------------------------------
st.title(" 🧠 Alzheimer's Disease Prediction APP by OLUWAMABAYOMIJE")
st.write("Predict the risk of Alzheimer's Disease based on patient features.")
st.sidebar.header("PATIENT DATA INPUTS")

def user_input_features():
    # ... (Input definitions remain the same) ...
    Age = st.sidebar.slider("Age", 60, 90, 70)
    Gender = st.sidebar.selectbox("Gender", [0,1], format_func=lambda x: "Male" if x==0 else "Female")
    # ... (Collect all inputs) ...
    Ethnicity = st.sidebar.selectbox("Ethnicity", [0,1,2,3], format_func=lambda x: ["Caucasian","African American","Asian","Other"][x])
    EducationLevel = st.sidebar.selectbox("Education Level", [0,1,2,3], format_func=lambda x: ["None","High School","Bachelor's","Higher"][x])
    BMI = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
    Smoking = st.sidebar.selectbox("Smoking", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    AlcoholConsumption = st.sidebar.slider("Alcohol Consumption (units/week)", 0, 20, 5)
    PhysicalActivity = st.sidebar.slider("Physical Activity (hours/week)", 0, 10, 3)
    DietQuality = st.sidebar.slider("Diet Quality (0-10)", 0, 10, 7)
    SleepQuality = st.sidebar.slider("Sleep Quality (4-10)", 4, 10, 7)
    FamilyHistoryAlzheimers = st.sidebar.selectbox("Family History of Alzheimer's", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    CardiovascularDisease = st.sidebar.selectbox("Cardiovascular Disease", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Diabetes = st.sidebar.selectbox("Diabetes", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Depression = st.sidebar.selectbox("Depression", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    HeadInjury = st.sidebar.selectbox("History of Head Injury", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Hypertension = st.sidebar.selectbox("Hypertension", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    SystolicBP = st.sidebar.slider("Systolic BP (mmHg)", 90, 180, 120)
    DiastolicBP = st.sidebar.slider("Diastolic BP (mmHg)", 60, 120, 80)
    CholesterolTotal = st.sidebar.slider("Total Cholesterol (mg/dL)", 150, 300, 200)
    CholesterolLDL = st.sidebar.slider("LDL Cholesterol (mg/dL)", 50, 200, 120)
    CholesterolHDL = st.sidebar.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
    CholesterolTriglycerides = st.sidebar.slider("Triglycerides (mg/dL)", 50, 400, 150)
    MMSE = st.sidebar.slider("MMSE Score (0-30)", 0, 30, 28)
    FunctionalAssessment = st.sidebar.slider("Functional Assessment (0-10)", 0, 10, 8)
    MemoryComplaints = st.sidebar.selectbox("Memory Complaints", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    BehavioralProblems = st.sidebar.selectbox("Behavioral Problems", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    ADL = st.sidebar.slider("Activities of Daily Living (0-10)", 0, 10, 9)
    Confusion = st.sidebar.selectbox("Confusion", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Disorientation = st.sidebar.selectbox("Disorientation", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    PersonalityChanges = st.sidebar.selectbox("Personality Changes", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    DifficultyCompletingTasks = st.sidebar.selectbox("Difficulty Completing Tasks", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Forgetfulness = st.sidebar.selectbox("Forgetfulness", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    
    # Assemble into a dictionary
    data = {
        # SCALED INPUTS (15 features)
        'Age': Age, 'BMI': BMI, 'AlcoholConsumption': AlcoholConsumption, 'PhysicalActivity': PhysicalActivity,
        'DietQuality': DietQuality, 'SleepQuality': SleepQuality, 'SystolicBP': SystolicBP, 'DiastolicBP': DiastolicBP,
        'CholesterolTotal': CholesterolTotal, 'CholesterolLDL': CholesterolLDL, 'CholesterolHDL': CholesterolHDL, 
        'CholesterolTriglycerides': CholesterolTriglycerides, 'MMSE': MMSE, 'FunctionalAssessment': FunctionalAssessment,
        'ADL': ADL, 
        
        # CATEGORICAL INPUTS (17 features)
        'EducationLevel': EducationLevel, 'Gender': Gender, 'Ethnicity': Ethnicity, 'Smoking': Smoking, 
        'FamilyHistoryAlzheimers': FamilyHistoryAlzheimers, 'CardiovascularDisease': CardiovascularDisease, 
        'Diabetes': Diabetes, 'Depression': Depression, 'HeadInjury': HeadInjury, 'Hypertension': Hypertension, 
        'MemoryComplaints': MemoryComplaints, 'BehavioralProblems': BehavioralProblems, 'Confusion': Confusion, 
        'Disorientation': Disorientation, 'PersonalityChanges': PersonalityChanges, 
        'DifficultyCompletingTasks': DifficultyCompletingTasks, 'Forgetfulness': Forgetfulness
    }
    
    # Create DataFrame in the CORRECT ORDER
    return pd.DataFrame(data, index=[0])[EXPECTED_COLUMNS_ORDER]

input_df = user_input_features()
st.sidebar.caption("© 2025 Geenord Tech Solutions")

# -------------------------------
# Prediction Button and Logic
# -------------------------------
st.subheader("Prediction")
if st.button("Predict Alzheimer's Risk"):
    
    # 1. Separate features based on required transformation
    df_scaled = input_df[SCALED_COLS].copy()
    df_cat = input_df[INT_CAT_COLS].copy()
    
    # 2. Apply the SCALER to the continuous features
    try:
        scaled_values = scaler.transform(df_scaled)
        df_scaled_transformed = pd.DataFrame(scaled_values, columns=SCALED_COLS, index=df_scaled.index)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.warning("The features used for scaling in the app may not match the features used to train the scaler.")
        st.stop()
        
    # 3. Cast Categorical Features to INT64
    for col in INT_CAT_COLS:
        df_cat[col] = df_cat[col].astype('int64') 

    # 4. Recombine into the final prediction DataFrame
    final_df = pd.concat([df_scaled_transformed, df_cat], axis=1)

    try:
        # Perform prediction on the correctly scaled and typed data
        prediction = model.predict(final_df)
        prediction_proba = model.predict_proba(final_df)

        st.success("✅ Prediction Complete")
        
        # Get the risk probability (for class 1: Alzheimer's)
        risk_probability = prediction_proba[0][1] * 100
        risk_class = 'Alzheimer' if prediction[0]==1 else 'No Alzheimer'
        
        # Determine color for the output
        if risk_probability >= 75:
            color = 'red'
        elif risk_probability >= 50:
            color = 'orange'
        else:
            color = 'green'

        
        # --- DISPLAY RESULTS ---
        st.markdown(f"**Predicted Class:** <span style='font-size: 24px; color:{color};'>**{risk_class}**</span>", unsafe_allow_html=True)
        st.markdown(f"**Prediction Probability:** **{risk_probability:.2f}%** risk")
        
        # --- CRITICAL WARNING AND RECOMMENDATION ---
        st.markdown("---")
        
        if risk_probability >= 50:
            st.error("🚨 HIGH RISK DETECTED!")
            st.markdown(
                """
                **Based on a risk probability of over 50%, it is highly recommended that you:**
                1. **Consult your physician or a specialist (Neurologist) immediately.**
                2. Share these input parameters and the prediction result with your doctor for further diagnostic evaluation.
                3. **Remember: This app is a screening tool, not a diagnosis.**
                """
            )
        elif risk_probability >= 20:
            st.warning("⚠️ Elevated Risk Noticed.")
            st.markdown(
                """
                The risk probability is elevated. It is a good practice to **discuss these results with your primary care physician** during your next check-up for early intervention and monitoring.
                """
            )
        else:
            st.info("✅ Low Risk Detected.")
            st.markdown(
                """
                The predicted risk is currently low. Continue focusing on healthy lifestyle choices and regular medical check-ups.
                """
            )
            
        st.markdown("---")
        
        # -------------------------------
        # Feature Importance (Uses final_df to get correct column names)
        # -------------------------------
        st.subheader("Feature Importance")
        importances = model.get_feature_importance()
        feature_names = final_df.columns
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10,8))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax)
        ax.set_title("Top 15 Feature Importances from CatBoost Model")
        st.pyplot(fig)

        st.subheader("🏆 Model Performance Summary")
        
        # Display the static scores
        st.markdown(f"""
        <div style='border: 1px solid #1f77b4; padding: 10px; border-radius: 5px; background-color: #e6f0ff;'>
            <h3 style='color: #1f77b4; margin-top: 0;'>Validation Metrics (Test Set)</h3>
            <p>🎯 Overall Accuracy: <span style='font-size: 1.2em; font-weight: bold;'>{ACCURACY*100:.2f}%</span></p>
            <p>📈 AUC Score: <span style='font-size: 1.2em; font-weight: bold;'>{AUC_SCORE*100:.2f}%</span></p>
            <p>🧪 Precision: <span style='font-size: 1.2em; font-weight: bold;'>{PRECISION*100:.2f}%</span></p>
            <p>🔎 Recall/Sensitivity: <span style='font-size: 1.2em; font-weight: bold;'>{RECALL*100:.2f}%</span></p>
            <p>🏆 F1-Score: <span style='font-size: 1.2em; font-weight: bold;'>{F1_SCORE*100:.2f}%</span></p>
            <p style='font-size: 0.9em; margin-bottom: 0;'> These scores reflect the model's performance on the unseen test data.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
    except Exception as e:
        st.error(f"A CatBoost error occurred: {e}")
        st.warning("An unexpected error occurred during prediction.")