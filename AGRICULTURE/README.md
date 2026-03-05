## **Soil Nutrient Analysis for Crop Prediction**

**Project Overview**

This project investigates how soil nutrient measurements influence crop selection using a machine learning approach. The analysis focuses on identifying which soil feature provides the strongest predictive signal when limited information about soil composition is available.

In many small-scale farming environments, farmers may not have access to full soil testing facilities. Instead, only one or a few soil measurements may be available. This project therefore examines whether a reliable crop prediction can be made using a single soil attribute.

Using a dataset containing soil nutrient levels and corresponding crop labels, the notebook explores the relationship between soil chemistry and crop suitability and evaluates which variable contributes most strongly to crop prediction.

**Dataset**

The dataset contains soil nutrient measurements associated with specific crop types.
Dataset size
2200 observations
5 variables

**Feature	Description**

N	Nitrogen content in the soil
P	Phosphorus content in the soil
K	Potassium content in the soil
ph	Soil acidity or alkalinity
crop = Crop type associated with the soil condition

The crop column represents the target variable used for classification.

**Problem Formulation**

The task is framed as a classification problem where soil attributes are used to predict the crop that can grow successfully under those conditions.
However, the notebook investigates a more constrained scenario:
If only one soil measurement can be obtained, which feature provides the most reliable crop prediction?
To answer this, the analysis evaluates each soil variable independently to determine its predictive power.

**Methodology**

The workflow implemented in the notebook consists of the following stages.

1. Data Inspection
The dataset is first loaded and inspected to understand:
structure of the dataset
number of observations and variables
feature types
class distribution of crops
Basic descriptive statistics are also examined to understand the range of soil measurements.

2. Feature–Target Separation

The soil variables (N, P, K, ph) are treated as input features, while the crop column serves as the prediction target.
Each feature is then evaluated individually to determine how informative it is for predicting crop type.

3. Model Training

For each soil variable, a classification model is trained using that single feature as input. This setup allows the analysis to compare how well each soil attribute performs independently.
The goal is not simply to build a predictive model but to identify the most informative soil measurement.

4. Model Evaluation

The performance of each model is evaluated to determine which feature provides the highest predictive accuracy.
By comparing the results across all four soil attributes, the analysis identifies which variable contributes the most information for crop prediction.

**Tools and Libraries**

The notebook was implemented using the following Python libraries:
Python
Pandas – data manipulation
Numpy – numerical computation
Scikit-learn – machine learning models
Matplotlib / Seaborn – data visualization
Jupyter Notebook – experimental environment

**Key Insights**

The central objective of the project is to determine which soil variable alone can best predict crop type.
This type of analysis can be useful in situations where:
full soil testing is not available
farmers rely on limited soil measurements
rapid agricultural recommendations are required
Understanding the predictive strength of individual soil attributes can help guide simplified decision-support tools for agriculture.

**AUTHOR**

B.Eng. Computer Engineering
- Focus: AI Solutions for Healthcare Challenges
- Expertise: Machine Learning, Data Science, Clinical Analytics
