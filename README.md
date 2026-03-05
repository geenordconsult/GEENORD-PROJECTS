# Early Detection of Alzheimer's Disease using Machine Learning

## Project Overview

This project develops a machine learning model for the **early detection of Alzheimer's disease**, one of the most prevalent forms of dementia affecting millions of adults worldwide. With approximately 70% of dementia cases being Alzheimer's disease and no definitive cure currently available, early detection is crucial for improving patient outcomes and quality of life.

The project leverages a **CatBoost classifier** to build a robust predictive model that achieves **95% accuracy** in identifying early-stage Alzheimer's disease based on clinical data and lifestyle factors.

## Research Motivation

As a researcher deeply invested in applying artificial intelligence to address critical health challenges, I identified dementia as a prominent issue requiring computational solutions. Recognizing the limitations of current diagnostic approaches, I developed this machine learning model to support healthcare professionals in early diagnosis and intervention.

## Key Findings

The analysis revealed that the following factors are most strongly associated with early-stage Alzheimer's disease:

- **Functional Assessments** - Brain activity and functional capacity measurements
- **Memory Complaints** - Patient-reported memory issues
- **Behavioral Problems** - Observable behavioral changes
- **Clinical Examination Scores** - Results from clinical assessments
- **Mini-Mental State Examination (MMSE)** - Standardized cognitive assessment scores

## Dataset Description

The project utilizes a synthetic dataset containing comprehensive health information for **2,149 patients** with IDs ranging from 4751 to 6900.

### Data Categories:

**Demographic Details:**
- Age (60-90 years)
- Gender (0: Male, 1: Female)
- Ethnicity (0: Caucasian, 1: African American, 2: Asian, 3: Other)
- Education Level (0: None, 1: High School, 2: Bachelor's, 3: Higher)

**Lifestyle Factors:**
- Body Mass Index (BMI) - 15 to 40
- Smoking Status (0: No, 1: Yes)
- Alcohol Consumption (0-20 units/week)
- Physical Activity (0-10 hours/week)
- Diet Quality (0-10 score)
- Sleep Quality (4-10 score)

**Medical History:**
- Family History of Alzheimer's
- Cardiovascular Disease
- Diabetes
- Depression
- Head Injury History
- Hypertension

**Clinical Measurements:**
- Systolic Blood Pressure (90-180 mmHg)
- Diastolic Blood Pressure (60-120 mmHg)
- Total Cholesterol (150-300 mg/dL)
- LDL Cholesterol (50-200 mg/dL)
- HDL Cholesterol (20-100 mg/dL)
- Triglycerides (50-400 mg/dL)

**Cognitive & Functional Assessments:**
- MMSE Score (0-30)
- Functional Assessment (0-10)
- Memory Complaints (0: No, 1: Yes)
- Behavioral Problems (0: No, 1: Yes)
- Activities of Daily Living (ADL) Score (0-10)

**Symptoms:**
- Confusion, Disorientation, Personality Changes
- Difficulty Completing Tasks
- Forgetfulness

**Target Variable:**
- Diagnosis (0: No Alzheimer's, 1: Alzheimer's Diagnosis)

## Project Structure

The notebook follows a comprehensive machine learning pipeline:

1. **Data Exploration** - Understanding dataset characteristics and distributions
2. **Data Cleaning** - Handling missing values and removing irrelevant columns
3. **Exploratory Data Analysis (EDA)** - Visualizing patterns and relationships
4. **Feature Engineering** - Creating and selecting relevant features
5. **Model Building** - Training the CatBoost classifier
6. **Model Evaluation** - Assessing performance with multiple metrics
7. **Deployment** - Streamlit web application for real-world usability

## Model Performance

- **Accuracy:** 95%
- **Algorithm:** CatBoost Classifier
- **Use Case:** Early detection support for healthcare professionals

## Deployment

The trained model has been deployed using **Streamlit**, a web application framework that enables real-world usability and accessibility for healthcare practitioners to make quick preliminary assessments.

## Contribution & Impact

**Role:** Sole Researcher - End-to-End Pipeline Development

This project demonstrates my ability to:
- Manage complete machine learning workflows from data preprocessing to deployment
- Apply advanced machine learning techniques to complex healthcare problems
- Identify and extract meaningful insights from clinical data
- Develop practical solutions that can support medical professionals

**Impact:** This project exemplifies how computational intelligence can be leveraged to tackle critical health challenges, supporting early diagnosis and ultimately improving patient care and outcomes.

## Installation & Requirements

### Dependencies:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
catboost
streamlit
```

### Setup:
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook Alzheimer.ipynb`
4. For web app: `streamlit run app.py`

## Usage

Open the Alzheimer.ipynb notebook to:
- Explore the full analysis pipeline
- Review data preprocessing and cleaning steps
- Understand feature importance and model performance
- Study the machine learning implementation

## Disclaimer

This dataset is **synthetic and generated for educational purposes**. It is ideal for data science and machine learning projects but should not be used for actual clinical diagnosis without validation against real clinical data.

## Future Enhancements

- Integration with real-world clinical datasets
- Model interpretability analysis (SHAP values)
- Cross-validation with additional demographic groups
- Real-time prediction dashboard

## Author

**Michael Jinadu (Geenord Consult tech Solutions)**
- Focus: AI Solutions for Healthcare Challenges
- Expertise: Machine Learning, Data Science, Clinical Analytics

## License

This project is part of the GEENORD-PROJECTS repository.

---

**Note:** Early detection of Alzheimer's disease can significantly improve patient outcomes. This project contributes to the ongoing effort to leverage technology in support of better healthcare outcomes.
