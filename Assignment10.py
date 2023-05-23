#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# URL of the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Define column names for the dataset
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Read the dataset into a DataFrame
df = pd.read_csv(url, header=None, names=column_names)

# Print the first few rows of the DataFrame
print(df.head())


# In[2]:


import matplotlib.pyplot as plt

# Plot histograms for each feature
df.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()


# In[3]:


import matplotlib.pyplot as plt

# Create box plots for each feature
df.boxplot(figsize=(8, 6))
plt.tight_layout()
plt.show()


# In[ ]:




