#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn import linear_model


# In[2]:


def read_data(file):
    return pd.read_csv(file)


# In[3]:


us = read_data('us_full.csv')


# In[4]:


# setup linear regression for orange production
orange_reg = linear_model.LinearRegression()
orange_reg.fit(us[['Temperature']], us['Oranges'])

# coefficient
orange_reg.coef_


# In[5]:


# intercept
orange_reg.intercept_


# In[6]:


# get correlation between temperature and orange production
us['Temperature'].corr(us['Oranges'])


# In[7]:


# plot graph of orange production vs average temperature
sns.set_theme()

orange_temp_graph = sns.lmplot(
    data=us,
    x="Temperature", y="Oranges",
    height=5
)

orange_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Oranges Produced (1000 MT)")


# In[ ]:




