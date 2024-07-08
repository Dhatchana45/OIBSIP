#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv("D:\\Unemployment in India.csv")
print(df.head())
print(df.columns)
df.columns = df.columns.str.strip()
print(df.describe())

print(df.isnull().sum())

df = df.fillna(method='ffill')
df['Date'] = df['Date'].str.strip()

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)


if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
else:
    print("Date column not found in the dataset")

plt.figure(figsize=(10, 6))
sns.histplot(df['Estimated Unemployment Rate (%)'], kde=True)
plt.title('Distribution of Estimated Unemployment Rate')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()



# Unemployment Rate by Region
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Region', y='Estimated Unemployment Rate (%)')
plt.title('Estimated Unemployment Rate by Region')
plt.xlabel('Region')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.xticks(rotation=90)
plt.show()

# Unemployment Rate over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('Estimated Unemployment Rate over Time by Region')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.legend(loc='upper right')
plt.show()

# Feature Engineering: Converting categorical variables to dummy variables
df = pd.get_dummies(df, columns=['Region', 'Frequency', 'Area'], drop_first=True)

# Define features and target
X = df.drop(columns=['Estimated Unemployment Rate (%)', 'Date'])
y = df['Estimated Unemployment Rate (%)']


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plotting actual vs predicted Unemployment Rates
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.xlabel('Actual Unemployment Rate (%)')
plt.ylabel('Predicted Unemployment Rate (%)')
plt.title('Actual vs Predicted Unemployment Rates')
plt.show()


# In[ ]:





# In[ ]:




