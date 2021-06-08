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


# setup linear regression for oat production
oat_reg = linear_model.LinearRegression()
oat_reg.fit(us[['Temperature']], us['Oats'])

# coefficient
oat_reg.coef_


# In[5]:


# intercept
oat_reg.intercept_


# In[6]:


# get correlation between temperature and oat production
us['Temperature'].corr(us['Oats'])


# In[7]:


# plot graph of oat production vs average temperature
sns.set_theme()

oat_temp_graph = sns.lmplot(
    data=us,
    x="Temperature", y="Oats",
    height=5
)

oat_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Oat Production (1000 MT)")


# In[ ]:




