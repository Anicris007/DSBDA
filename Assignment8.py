#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns

# Load the Titanic dataset from Seaborn
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic.head())


# In[2]:


# Bar plot of passenger survival
sns.countplot(x='survived', data=titanic)

# Bar plot of passenger survival based on gender
sns.countplot(x='survived', hue='sex', data=titanic)

# Scatter plot of passenger age and fare
sns.scatterplot(x='age', y='fare', data=titanic)

# Box plot of passenger class and fare
sns.boxplot(x='pclass', y='fare', data=titanic)


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from Seaborn
titanic = sns.load_dataset('titanic')

# Plot a histogram of ticket prices
sns.histplot(data=titanic, x='fare', kde=True)

# Display the plot
plt.show()


# In[ ]:




