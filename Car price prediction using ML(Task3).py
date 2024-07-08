#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


df = pd.read_csv("D:\\car data.csv")

# Perform basic EDA
print(df.head())
print(df.info())
print(df.describe())

# Visualize the data
sns.pairplot(df)
plt.show()

# Check for missing values
print(df.isnull().sum())


# In[7]:


# Define features and target
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']

# One-hot encode categorical features
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
numeric_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner']

# Create a preprocessor for the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a pipeline for the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=0))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[12]:


# Example of predicting a new car's selling price
new_data = pd.DataFrame({
    'Year': [2015],
    'Present_Price': [8.5],
    'Driven_kms': [30000],
    'Fuel_Type': ['Petrol'],
    'Selling_type': ['Dealer'],
    'Transmission': ['Manual'],
    'Owner': [0]
})

# Predict the selling price
predicted_price = model.predict(new_data)
print(f'Predicted Selling Price: {predicted_price[0]}')


# In[ ]:




