#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn import linear_model


# In[2]:


def read_data(file):
    return pd.read_csv(file)


# In[4]:


miami = read_data('miami_full.csv')


# In[6]:


# setup linear regression for lemon production
lemon_reg = linear_model.LinearRegression()
lemon_reg.fit(miami[['Average Temperature', 'Average Rainfall']], miami['Lemons'])

# coefficients (lemons)
lemon_reg.coef_


# In[8]:


lemon_reg.intercept_


# In[9]:


# get correlation between temperature and lemons produced
miami['Average Temperature'].corr(miami['Lemons'])


# In[11]:


# get correlation bewteen rainfall and lemons produced
miami['Average Rainfall'].corr(miami['Lemons'])


# In[13]:


# plot graph of lemon production vs average temperature
sns.set_theme()

lemon_temp_graph = sns.lmplot(
    data=miami,
    x="Average Temperature", y="Lemons",
    height=5
)

lemon_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Lemons Produced (1000 MT)")


# In[14]:


# plot graph of lemon production vs average rainfall
lemon_rain_graph = sns.lmplot(
    data=miami,
    x="Average Rainfall", y="Lemons",
    height=5
)

lemon_rain_graph.set_axis_labels("Average Rainfall (inches)", "Lemons Produced (1000 MT)")

