#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


df = pd.read_csv("iris.csv")
df.head()


# In[6]:


df.insult()


# In[7]:


df.isnull()


# In[8]:


df.describe()


# In[9]:


grouped = df.groupby('sepal_length')


# In[10]:


mean_values = grouped['numerical_column'].mean()


# In[11]:


mean_values = grouped['sepal_length'].mean()


# In[12]:


print(grouped)


# In[13]:


for group_name, group_data in grouped:
    print("Group Name:", group_name)
    print(group_data)
    print("\n")  


# In[15]:


data_types = df.dtypes


# In[16]:


print(data_types)


# In[17]:


print(df.dtypes)


# In[ ]:




