import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

# -------------------------------
# Load CatBoost model
# -------------------------------
@st.cache_data
def load_model():
    model = CatBoostClassifier()
    # Ensure this path is correct for your machine
   model.load_model("Alzheimer_model.cbm") 
    return model

model = load_model()

# -------------------------------
# App title
# -------------------------------
st.title("Alzheimer's Disease Prediction App 🧠 by OLUWAMABAYOMIJE")
st.write("Evaluate a patient's risk for developing Alzheimer's Disease.")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Patient Data Inputs")

def user_input_features():
    # Demographics
    Age = st.sidebar.slider("Age", 60, 90, 70)
    Gender = st.sidebar.selectbox("Gender", [0,1], format_func=lambda x: "Male" if x==0 else "Female")
    Ethnicity = st.sidebar.selectbox("Ethnicity", [0,1,2,3], 
                                     format_func=lambda x: ["Caucasian","African American","Asian","Other"][x])
    EducationLevel = st.sidebar.selectbox("Education Level", [0,1,2,3], 
                                          format_func=lambda x: ["None","High School","Bachelor's","Higher"][x])

    # Lifestyle Factors
    BMI = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
    Smoking = st.sidebar.selectbox("Smoking", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    AlcoholConsumption = st.sidebar.slider("Alcohol Consumption (units/week)", 0.0, 20.0, 5.0)
    PhysicalActivity = st.sidebar.slider("Physical Activity (hours/week)", 0.0, 10.0, 3.0)
    DietQuality = st.sidebar.slider("Diet Quality (0-10)", 0.0, 10.0, 7.0)
    SleepQuality = st.sidebar.slider("Sleep Quality (4-10)", 4.0, 10.0, 7.0)

    # Medical History
    FamilyHistoryAlzheimers = st.sidebar.selectbox("FamilyHistoryAlzheimers", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    CardiovascularDisease = st.sidebar.selectbox("CardiovascularDisease", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Diabetes = st.sidebar.selectbox("Diabetes", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Depression = st.sidebar.selectbox("Depression", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    HeadInjury = st.sidebar.selectbox("HeadInjury", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Hypertension = st.sidebar.selectbox("Hypertension", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

    # Clinical Measurements
    SystolicBP = st.sidebar.slider("Systolic BP (mmHg)", 90, 180, 120)
    DiastolicBP = st.sidebar.slider("Diastolic BP (mmHg)", 60, 120, 80)
    CholesterolTotal = st.sidebar.slider("Total Cholesterol (mg/dL)", 150.0, 300.0, 200.0)
    CholesterolLDL = st.sidebar.slider("LDL Cholesterol (mg/dL)", 50.0, 200.0, 120.0)
    CholesterolHDL = st.sidebar.slider("HDL Cholesterol (mg/dL)", 20.0, 100.0, 50.0)
    CholesterolTriglycerides = st.sidebar.slider("Triglycerides (mg/dL)", 50.0, 400.0, 150.0)

    # Cognitive & Functional Assessments
    MMSE = st.sidebar.slider("MMSE Score (0-30)", 0.0, 30.0, 28.0)
    FunctionalAssessment = st.sidebar.slider("Functional Assessment (0-10)", 0.0, 10.0, 8.0)
    MemoryComplaints = st.sidebar.selectbox("MemoryComplaints", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    BehavioralProblems = st.sidebar.selectbox("BehavioralProblems", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    ADL = st.sidebar.slider("Activities of Daily Living (0-10)", 0.0, 10.0, 9.0)

    # Symptoms
    Confusion = st.sidebar.selectbox("Confusion", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Disorientation = st.sidebar.selectbox("Disorientation", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    PersonalityChanges = st.sidebar.selectbox("PersonalityChanges", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    DifficultyCompletingTasks = st.sidebar.selectbox("DifficultyCompletingTasks", [0,1], 
                                                     format_func=lambda x: "No" if x==0 else "Yes")
    Forgetfulness = st.sidebar.selectbox("Forgetfulness", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

    # Pack data
    data = {
        'Age': Age,
        'Gender': Gender,
        'Ethnicity': Ethnicity,
        'EducationLevel': EducationLevel,
        'BMI': BMI,
        'Smoking': Smoking,
        'AlcoholConsumption': AlcoholConsumption,
        'PhysicalActivity': PhysicalActivity,
        'DietQuality': DietQuality,
        'SleepQuality': SleepQuality,
        'FamilyHistoryAlzheimers': FamilyHistoryAlzheimers,
        'CardiovascularDisease': CardiovascularDisease,
        'Diabetes': Diabetes,
        'Depression': Depression,
        'HeadInjury': HeadInjury,
        'Hypertension': Hypertension,
        'SystolicBP': SystolicBP,
        'DiastolicBP': DiastolicBP,
        'CholesterolTotal': CholesterolTotal,
        'CholesterolLDL': CholesterolLDL,
        'CholesterolHDL': CholesterolHDL,
        'CholesterolTriglycerides': CholesterolTriglycerides,
        'MMSE': MMSE,
        'FunctionalAssessment': FunctionalAssessment,
        'MemoryComplaints': MemoryComplaints,
        'BehavioralProblems': BehavioralProblems,
        'ADL': ADL,
        'Confusion': Confusion,
        'Disorientation': Disorientation,
        'PersonalityChanges': PersonalityChanges,
        'DifficultyCompletingTasks': DifficultyCompletingTasks,
        'Forgetfulness': Forgetfulness
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# -------------------------------
# Correct types
# -------------------------------
int_features = ['Age','SystolicBP','DiastolicBP']
float_features = [
    'BMI','AlcoholConsumption','PhysicalActivity','DietQuality','SleepQuality',
    'CholesterolTotal','CholesterolLDL','CholesterolHDL','CholesterolTriglycerides',
    'MMSE','FunctionalAssessment','ADL'
]
cat_features = [
    'Gender','Ethnicity','EducationLevel','Smoking','FamilyHistoryAlzheimers',
    'CardiovascularDisease','Diabetes','Depression','HeadInjury','Hypertension',
    'MemoryComplaints','BehavioralProblems','Confusion','Disorientation',
    'PersonalityChanges','DifficultyCompletingTasks','Forgetfulness'
]

# Apply Type Conversions
input_df[int_features] = input_df[int_features].astype(int)
input_df[float_features] = input_df[float_features].astype(float)
input_df[cat_features] = input_df[cat_features].astype(str)

# ==========================================
# CRITICAL FIX: REORDER COLUMNS
# ==========================================
# Get the feature names directly from the model object
# This ensures the input_df matches the training order exactly
try:
    expected_order = model.feature_names_
    if expected_order:
        input_df = input_df[expected_order]
except Exception as e:
    st.error(f"Could not retrieve feature names from model. Please ensure columns match training data exactly. Error: {e}")

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("**PREDICTION RESULTS**")
        st.write(f"**Predicted Class:** {'Alzheimer' if prediction[0]==1 else 'No Alzheimer'}")
        st.write(f"**Prediction Probability:** {prediction_proba[0][1]*100:.2f}% risk")

        # -------------------------------
        # Feature Importance
        # -------------------------------
        st.subheader("Feature Importance")
        importances = model.get_feature_importance()
        
        # Use the COLUMNS from the DATAFRAME (which are now sorted correctly)
        feature_names = input_df.columns 

        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10,8))
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        st.pyplot(fig)
        
    except Exception as e:

        st.error(f"An error occurred during prediction: {e}")
