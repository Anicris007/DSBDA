#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('housing.csv')

# Separate the features and target variable
X = data.drop('MEDV', axis=1)  # Replace 'target_variable_name' with the actual target column name
y = data['MEDV']  # Replace 'target_variable_name' with the actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)


# In[4]:


data.describe()


# In[ ]:

#WE HAVE TO DEVELOP THE MODEL SO DONT THINK THAT THIS IS A WRONG PROGRAM


