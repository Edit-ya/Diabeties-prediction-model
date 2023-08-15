#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from PIL import Image

# In[12]:





# In[13]:



# In[14]:




# Load the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('E:/diabeties model/diabetes.csv')

# Separate the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# Load the Streamlit app
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(diabetes_dataset.describe())

# Define the model
classifier = svm.SVC()

# Define the parameter grid for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Create the GridSearchCV object
grid_search = GridSearchCV(classifier, param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, Y_train)

# Train the model with the best parameters
classifier = grid_search.best_estimator_
classifier.fit(X_train, Y_train)

# ... (Rest of the code remains the same)

# User input function
def user_report():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    SkinThickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    Insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    Age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
        'Age': Age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


# Get user input
user_data = user_report()
user_data_scaled = scaler.transform(user_data)

# Make prediction
user_result = classifier.predict(user_data_scaled)
print(user_result)
# Display prediction
st.subheader('Your Report:')
if user_result[0] == 0:
    st.write('You are not Diabetic')
else:
    st.write('You are Diabetic')

# Display accuracy
accuracy = accuracy_score(Y_test, classifier.predict(X_test)) * 100
st.subheader('Model Accuracy:')
st.write(f'Accuracy: {accuracy:.2f}%')


# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = diabetes_dataset, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_Glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = diabetes_dataset, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Glucose)



# Age vs BloodPressure
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_BloodPressure = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = diabetes_dataset, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_BloodPressure)


# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = diabetes_dataset, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = diabetes_dataset, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# # Age vs BMI
# st.header('BMI Value Graph (Others vs Yours)')
# fig_bmi = plt.figure()
# ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = diabetes_dataset, hue = 'Outcome', palette='rainbow')
# ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['bmi'], s = 150, color = color)
# plt.xticks(np.arange(10,100,5))
# plt.yticks(np.arange(0,70,5))
# plt.title('0 - Healthy & 1 - Unhealthy')
# st.pyplot(fig_bmi)


# Age vs DiabetesPedigreeFunction
st.header('DPF Value Graph (Others vs Yours)')
fig_DiabetesPedigreeFunction = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = diabetes_dataset, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['DiabetesPedigreeFunction'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_DiabetesPedigreeFunction)

# Output
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)

