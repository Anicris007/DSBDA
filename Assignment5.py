#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
data = pd.read_csv('Social_Network_Ads.csv')

# Split the dataset into features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode categorical variables
label_encoder = LabelEncoder()
X[:, 1] = label_encoder.fit_transform(X[:, 1])  # Encoding gender column

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)  # One-hot encoding the country column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[3]:


from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract true positive (TP), false positive (FP), true negative (TN), and false negative (FN)
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

# Calculate accuracy
accuracy = (TP + TN) / (TP + FP + TN + FN)
print("Accuracy:", accuracy)

# Calculate error rate
error_rate = (FP + FN) / (TP + FP + TN + FN)
print("Error Rate:", error_rate)

# Calculate precision
precision = TP / (TP + FP)
print("Precision:", precision)

# Calculate recall (also known as sensitivity or true positive rate)
recall = TP / (TP + FN)
print("Recall:", recall)


# In[ ]:




