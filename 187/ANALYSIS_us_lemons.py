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


# setup linear regression for lemon production
lemon_reg = linear_model.LinearRegression()
lemon_reg.fit(us[['Temperature']], us['Lemons'])

# coefficient
lemon_reg.coef_


# In[5]:


# intercept
lemon_reg.intercept_


# In[6]:


# get correlation between temperature and lemon production
us['Temperature'].corr(us['Lemons'])


# In[8]:


# plot graph of lemon production vs average temperature
sns.set_theme()

lemon_temp_graph = sns.lmplot(
    data=us,
    x="Temperature", y="Lemons",
    height=5
)

lemon_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Lemons Produced (1000 MT)")


# In[ ]:




