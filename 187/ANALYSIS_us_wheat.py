#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
from sklearn import linear_model


# In[3]:


def read_data(file):
    return pd.read_csv(file)


# In[9]:


us = read_data('us_full.csv')


# In[10]:


# setup linear regression for wheat production
wheat_reg = linear_model.LinearRegression()
wheat_reg.fit(us[['Temperature']], us['Wheat'])

# coefficient
wheat_reg.coef_


# In[11]:


# intercept
wheat_reg.intercept_


# In[12]:


# get correlation between temperature and wheat production
us['Temperature'].corr(us['Wheat'])


# In[13]:


# plot graph of wheat production vs temperature
sns.set_theme()

wheat_temp_graph = sns.lmplot(
    data=us,
    x="Temperature", y="Wheat",
    height=5
)

wheat_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Wheat Production (1000 MT)")


# In[ ]:




