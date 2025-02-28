import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import os

# Load the model
model = joblib.load('titanic_model.pkl')

# Streamlit app
st.title('Titanic Survival Prediction')

# Input features
st.header('Passenger Information')
passengerid = st.number_input('PassengerId', min_value=1, value=892) # Add PassengerId input
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, value=10.0)
embarked_c = st.checkbox('Embarked from Cherbourg (C)')
embarked_q = st.checkbox('Embarked from Queenstown (Q)')
embarked_s = st.checkbox('Embarked from Southampton (S)')


# Create input dataframe
input_data = pd.DataFrame({
    'PassengerId': [passengerid],  
    'Pclass': [pclass],
    'Sex_male': [1 if sex == 'male' else 0],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_Q': [1 if embarked_q else 0], 
    'Embarked_S': [1 if embarked_s else 0]  
}, index=[0])

# Get the feature names from the trained model
feature_names = model.feature_names_in_  # Access feature names

# Reorder columns to match training data using feature_names
input_data = input_data[feature_names]  # Use feature_names to reorder

# Make prediction
prediction = model.predict(input_data)

# Display prediction
st.header('Prediction')
if prediction[0] == 1:
    st.write('Passenger is predicted to survive.')
else:
    st.write('Passenger is predicted to not survive.')
