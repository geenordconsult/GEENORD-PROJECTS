# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

# -------------------------------
# Load CatBoost model
# -------------------------------
@st.cache_data
def load_model():
    model = CatBoostClassifier()
    model.load_model("alzheimer_model.cbm")  # replace with your model path
    return model

model = load_model()

# -------------------------------
# App title
# -------------------------------
st.title("Alzheimer's Disease Prediction App 🧠")
st.write("Predict the risk of Alzheimer's Disease based on patient features.")

# -------------------------------
# Sidebar inputs
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
    AlcoholConsumption = st.sidebar.slider("Alcohol Consumption (units/week)", 0, 20, 5)
    PhysicalActivity = st.sidebar.slider("Physical Activity (hours/week)", 0, 10, 3)
    DietQuality = st.sidebar.slider("Diet Quality (0-10)", 0, 10, 7)
    SleepQuality = st.sidebar.slider("Sleep Quality (4-10)", 4, 10, 7)
    
    # Medical History
    FamilyHistoryAlzheimers = st.sidebar.selectbox("Family History of Alzheimer's", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    CardiovascularDisease = st.sidebar.selectbox("Cardiovascular Disease", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Diabetes = st.sidebar.selectbox("Diabetes", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Depression = st.sidebar.selectbox("Depression", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    HeadInjury = st.sidebar.selectbox("History of Head Injury", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Hypertension = st.sidebar.selectbox("Hypertension", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    
    # Clinical Measurements
    SystolicBP = st.sidebar.slider("Systolic BP (mmHg)", 90, 180, 120).dtype('float')
    DiastolicBP = st.sidebar.slider("Diastolic BP (mmHg)", 60, 120, 80)
    CholesterolTotal = st.sidebar.slider("Total Cholesterol (mg/dL)", 150, 300, 200)
    CholesterolLDL = st.sidebar.slider("LDL Cholesterol (mg/dL)", 50, 200, 120)
    CholesterolHDL = st.sidebar.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
    CholesterolTriglycerides = st.sidebar.slider("Triglycerides (mg/dL)", 50, 400, 150)
    
    # Cognitive and Functional Assessments
    MMSE = st.sidebar.slider("MMSE Score (0-30)", 0, 30, 28)
    FunctionalAssessment = st.sidebar.slider("Functional Assessment (0-10)", 0, 10, 8)
    MemoryComplaints = st.sidebar.selectbox("Memory Complaints", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    BehavioralProblems = st.sidebar.selectbox("Behavioral Problems", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    ADL = st.sidebar.slider("Activities of Daily Living (0-10)", 0, 10, 9)
    
    # Symptoms
    Confusion = st.sidebar.selectbox("Confusion", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Disorientation = st.sidebar.selectbox("Disorientation", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    PersonalityChanges = st.sidebar.selectbox("Personality Changes", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    DifficultyCompletingTasks = st.sidebar.selectbox("Difficulty Completing Tasks", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    Forgetfulness = st.sidebar.selectbox("Forgetfulness", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    
    # Assemble into DataFrame
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
# Prediction
# -------------------------------
st.subheader("Prediction")
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.write(f"**Predicted Class:** {'Alzheimer' if prediction[0]==1 else 'No Alzheimer'}")
st.write(f"**Prediction Probability:** {prediction_proba[0][1]*100:.2f}% risk")

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("Feature Importance")
importances = model.get_feature_importance()
feature_names = input_df.columns
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax)
st.pyplot(fig)

# -------------------------------
# Optional: Classification Report
# -------------------------------
st.subheader("Sample Classification Report")
try:
    X_test = pd.read_csv("X_test.csv")  # replace with your test set
    y_test = pd.read_csv("y_test.csv")['target']  # ensure correct column name
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'))
except:
    st.write("Test data not found. You can skip this section or load your test set.")


