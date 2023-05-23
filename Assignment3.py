#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Read the Iris dataset (assuming it's in the same directory as this script)
data = pd.read_csv('iris.csv')

# Group the data by species and calculate summary statistics
grouped_stats = data.groupby('species')['sepal_length', 'sepal_width', 'petal_length', 'petal_width'].agg(['mean', 'median', 'min', 'max', 'std'])
# Create a list of species
species_list = grouped_stats.index.tolist()

# Print the summary statistics
print(grouped_stats)
print("Species List:", species_list)






