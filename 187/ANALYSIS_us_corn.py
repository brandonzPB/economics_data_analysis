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


# setup linear regression for corn production
corn_reg = linear_model.LinearRegression()
corn_reg.fit(us[['Temperature']], us['Corn'])

# coefficient
corn_reg.coef_


# In[5]:


# intercept
corn_reg.intercept_


# In[6]:


# get correlation between temperature and corn production
us['Temperature'].corr(us['Corn'])


# In[7]:


# plot graph of corn production vs average temperature
sns.set_theme()

corn_temp_graph = sns.lmplot(
    data=us,
    x="Temperature", y="Corn",
    height=5
)

corn_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Corn Production (1000 MT)")

