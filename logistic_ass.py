# -*- coding: utf-8 -*-
"""Logistic_Ass.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Xr369GK-ek448zS2laUMV4u5qg_EVc8B
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
sns.set_theme(style='darkgrid',palette='rainbow')
from sklearn.model_selection import train_test_split

df_train= pd.read_csv('Titanic_train.csv')
df_train

df_test= pd.read_csv('Titanic_test.csv')
df_test

# Examine the features, their types, and summary statistics.

df_train.info()

# Generate descriptive statistics
df_train.describe(include='all')

# Create visualizations such as histograms, box plots, or pair plots to visualize the distributions and relationships between features.

import matplotlib.pyplot as plt
# Histograms for numerical features
plt.figure(figsize=(10, 6))
df_train['Age'].hist(bins=20)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
df_train['Fare'].hist(bins=20)
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Box plots for numerical features
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=df_train)
plt.title('Age Distribution by Survival')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=df_train)
plt.title('Fare Distribution by Survival')
plt.show()

# Pair plot for selected numerical features
sns.pairplot(df_train[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.show()

# Count plots for categorical features
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', hue='Survived', data=df_train)
plt.title('Survival by Sex')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Survived', data=df_train)
plt.title('Survival by Passenger Class')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked', hue='Survived', data=df_train)
plt.title('Survival by Embarked')
plt.show()

# Analyze any patterns or correlations observed in the data.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (assuming it's already loaded in df_train)
# df_train = pd.read_csv('Titanic_train.csv')

# Analyze correlations using a heatmap
# Select only numerical features for correlation analysis
numerical_features = df_train.select_dtypes(include=['number'])  # Select numerical columns
plt.figure(figsize=(12, 8))
correlation_matrix = numerical_features.corr() # Calculate correlation for numerical features only
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Analyze survival rate by passenger class and sex
print(df_train.groupby(['Pclass', 'Sex'])['Survived'].mean())

# Analyze survival rate by age groups
# Define age groups
bins = [0, 18, 30, 50, 100]  # Adjust bins as needed
labels = ['0-18', '19-30', '31-50', '51+']
df_train['AgeGroup'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=False)

# Analyze survival rate by age group
print(df_train.groupby('AgeGroup')['Survived'].mean())

# Analyze survival rate by fare groups
# Define fare groups (using quantiles for more robust grouping)
fare_quantiles = df_train['Fare'].quantile([0, 0.25, 0.5, 0.75, 1])
df_train['FareGroup'] = pd.cut(df_train['Fare'], bins=fare_quantiles, labels=['Low', 'Medium-Low', 'Medium', 'High'], include_lowest=True)
print(df_train.groupby('FareGroup')['Survived'].mean())

# Further exploration of SibSp and Parch
# Combine SibSp and Parch into a new family size feature
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1

# Analyze survival rate by family size
print(df_train.groupby('FamilySize')['Survived'].mean())

# Visualize the distribution of family sizes
plt.figure(figsize=(10, 6))
sns.countplot(x='FamilySize', hue='Survived', data=df_train)
plt.title('Survival Rate by Family Size')
plt.show()

# a. Handle missing values (e.g., imputation).

# Fill missing 'Age' values with the median age
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the most frequent value
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Embarked'].fillna(df_test['Embarked'].mode()[0], inplace=True)

# Fill missing 'Fare' values with the median fare
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)

# Encode categorical variables.

import pandas as pd
# Ensure 'AgeGroup' and 'FareGroup' are in df_train before applying get_dummies
# If you've run the analysis in a previous cell, re-run that cell first to create these columns
# or re-create them here

# Define age groups
bins = [0, 18, 30, 50, 100]  # Adjust bins as needed
labels = ['0-18', '19-30', '31-50', '51+']
df_train['AgeGroup'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=False)

# Define fare groups (using quantiles for more robust grouping)
fare_quantiles = df_train['Fare'].quantile([0, 0.25, 0.5, 0.75, 1])
df_train['FareGroup'] = pd.cut(df_train['Fare'], bins=fare_quantiles, labels=['Low', 'Medium-Low', 'Medium', 'High'], include_lowest=True)

# Now proceed with get_dummies
df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked', 'Pclass', 'AgeGroup', 'FareGroup'], drop_first=True)

# Reload df_test to ensure it has the original columns
df_test = pd.read_csv('Titanic_test.csv')  # Reload df_test
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_test['Embarked'].fillna(df_test['Embarked'].mode()[0], inplace=True)
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)



df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True) #Note: AgeGroup and FareGroup are not present in the test data

# Display the updated DataFrame
print(df_train.head())
print(df_test.head())

# Build a logistic regression model using appropriate libraries (e.g., scikit-learn).

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define features (X) and target (y)
X = df_train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df_train['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Train the model using the training data.

import pandas as pd
# The model is already trained in the provided code.
# The following lines are for prediction on the test dataset.

# Prepare the test data
X_test_final = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Ensure the test data has the same columns as the training data
missing_cols = set(X.columns) - set(X_test_final.columns)
for c in missing_cols:
    X_test_final[c] = 0
X_test_final = X_test_final[X.columns]


# Make predictions on the final test set
y_pred_final = model.predict(X_test_final)

# Create a submission DataFrame
submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred_final})

# Save the predictions to a CSV file
submission.to_csv('titanic_submission.csv', index=False)

# Evaluate the performance of the model on the testing data using accuracy, precision, recall, F1-score, and ROC-AUC score.
# Visualize the ROC curve.

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Assuming y_test and y_pred are already defined from the previous code.

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Predict probabilities for ROC AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"ROC-AUC score: {roc_auc}")

# Visualize the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Interpret the coefficients of the logistic regression model.

import pandas as pd
# Access the coefficients and intercept
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# Create a DataFrame for better visualization
feature_names = X.columns
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Print the coefficients
print("Coefficients:")
print(coefficients_df)
print(f"\nIntercept: {intercept}")

# Interpretation:
print("\nInterpretation of Coefficients:")
for index, row in coefficients_df.iterrows():
    feature = row['Feature']
    coefficient = row['Coefficient']
    print(f"- {feature}:")
    if coefficient > 0:
        print(f"  An increase in {feature} is associated with an increase in the log-odds of survival.")
        print(f"  Specifically, a one-unit increase in {feature} increases the log-odds of survival by {coefficient:.2f}.")
    else:
        print(f"  An increase in {feature} is associated with a decrease in the log-odds of survival.")
        print(f"  Specifically, a one-unit increase in {feature} decreases the log-odds of survival by {-coefficient:.2f}.")
    print("-"*20)

# Note on categorical features:
print("For categorical features (represented as dummy variables):")
print("- A coefficient represents the change in log-odds of survival compared to the reference category.")
print("- For example, Sex_male represents the difference in log-odds between males and the reference category (females).")

# Note on interpretation:
print("\nImportant Note on Interpretation:")
print("- These are log-odds, not probabilities. To get probabilities, you need to apply the logistic function (sigmoid).")
print("- The magnitude of the coefficient indicates the strength of the relationship between a feature and the outcome.")
print("- Positive coefficients indicate that higher values of the feature are associated with higher probabilities of survival, and vice-versa.")

# prompt: Discuss the significance of features in predicting the target variable (survival probability in this case).

# Feature Significance Discussion based on the provided code:

# The code performs exploratory data analysis (EDA) and builds a logistic regression model to predict survival on the Titanic.  Let's discuss feature significance based on the analysis:

# 1. Numerical Features:
#    - Age: The box plots and the pairplot suggest a potential relationship between age and survival.  Older passengers may have had a lower survival rate.  The model coefficients will quantify this.  The creation of 'AgeGroup' suggests that a non-linear relationship might be present (binning into groups).
#    - Fare:  Similar to age, fare seems to correlate with survival, possibly indicating that wealthier passengers had better access to lifeboats. Again, the model will quantify this relationship.  Grouping fares using quantiles into 'FareGroup' suggests a non-linearity to explore.
#    - Pclass: The survival rate analysis by passenger class clearly shows that first-class passengers had a much higher survival rate than those in third class.  This would be a very significant predictor.  Note that this is a categorical variable encoded into numerical dummies.
#    - FamilySize (engineered): This feature, combining 'SibSp' and 'Parch', might indicate survival chances related to family size.  The analysis shows a relationship.  Extreme values of family size (very large or very small) may have implications for survival.

# 2. Categorical Features:
#    - Sex: This is likely a highly significant feature, as shown by the count plots.  Females had a dramatically higher survival rate than males.
#    - Embarked: The port of embarkation might have had a minor influence, but the differences seem less pronounced than sex or passenger class.
#    - AgeGroup and FareGroup (engineered): These categorical features are created by binning continuous variables, potentially capturing non-linear effects.

# 3. Model Coefficients:
#   The code provides code to print coefficients.  Positive coefficients indicate that an increase in the feature is associated with an increase in the log-odds of survival (and thus, probability of survival).  Negative coefficients represent the opposite.  The magnitude of the coefficient reflects the strength of the relationship.  Careful interpretation is crucial due to the logarithmic relationship.  Also, for categorical features, the reference category must be considered; the coefficients represent the differences from the reference category.

# 4. Correlation Analysis:
#    - Correlation among numerical features should be considered. High correlation between two features can lead to multicollinearity, which can complicate the interpretation of coefficients.

# 5. Feature Engineering:
#    - The creation of 'AgeGroup', 'FareGroup', and 'FamilySize' is an attempt to improve model performance. These transformed features might capture relationships between the original features and the target variable more effectively than the raw variables alone.

# Overall Significance:
# Based on the EDA and the likely coefficients, the most significant features in predicting survival are likely to be 'Sex', 'Pclass', 'Fare', and 'Age'. The engineered 'AgeGroup', 'FareGroup', and 'FamilySize' features will also show importance. 'Embarked' is less influential. The coefficients of the logistic regression model provides quantitative measure of the importance of each feature.

#!pip install streamlit

# In this task, you will deploy your logistic regression model using Streamlit. The deployment can be done locally or online via Streamlit Share.
# Your task includes creating a Streamlit app in Python that involves loading your trained model and setting up user inputs for predictions. 


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the trained model (replace 'your_model.pkl' with the actual filename)
# Assuming you've saved your trained model as 'model' in the previous code
# Save the trained model
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the saved model
loaded_model = pickle.load(open(filename, 'rb'))


# Load the data (replace 'your_data.csv' with the actual filename)
# Assuming your test data is named 'df_test' and has been processed as in your notebook
df_test = pd.read_csv('Titanic_test.csv') # Reload df_test to ensure it has the original columns
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_test['Embarked'].fillna(df_test['Embarked'].mode()[0], inplace=True)
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)
df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# Create the Streamlit app
st.title("Titanic Survival Prediction")

# Create input fields for user data
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked_c = st.checkbox("Embarked from Cherbourg")
embarked_q = st.checkbox("Embarked from Queenstown")


# Create a button to trigger prediction
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked_Q': [1 if embarked_q else 0],
        'Embarked_S': [0 if embarked_c or embarked_q else 1], #Added Embarked_S
        'Embarked_C': [1 if embarked_c else 0],
        'Pclass_2' : [0],
        'Pclass_3' : [0] if pclass==1 else (1 if pclass==3 else 0),
        'Sex_male': [1 if sex=="male" else 0]
    })


    # Ensure the test data has the same columns as the training data
    missing_cols = set(X.columns) - set(input_data.columns)
    for c in missing_cols:
      input_data[c] = 0
    input_data = input_data[X.columns]

    # Make prediction
    prediction = loaded_model.predict(input_data)[0]

    # Display the prediction
    if prediction == 1:
        st.write("Passenger is likely to survive.")
    else:
        st.write("Passenger is likely to not survive.")

#!streamlit run /usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py



