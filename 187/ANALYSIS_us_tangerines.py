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


# setup linear regression of tangerine production
tang_reg = linear_model.LinearRegression()
tang_reg.fit(us[['Temperature']], us['Tangerines'])

# coefficient
tang_reg.coef_


# In[5]:


# intercept
tang_reg.intercept_


# In[6]:


# get correlation between temperature and tangerine production
us['Temperature'].corr(us['Tangerines'])


# In[7]:


# plot graph of tangerine production vs average temperature
sns.set_theme()

tang_temp_graph = sns.lmplot(
    data=us,
    x="Temperature", y="Tangerines",
    height=5
)

tang_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Tangerines Produced (1000 MT)")


# In[ ]:




