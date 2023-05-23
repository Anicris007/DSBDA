#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from Seaborn
titanic = sns.load_dataset('titanic')

# Plot a box plot of age distribution with respect to gender and survival
sns.boxplot(x='sex', y='age', hue='survived', data=titanic)

# Add a title to the plot
plt.title('Age Distribution by Gender and Survival')

# Display the plot
plt.show()


# In[ ]:




