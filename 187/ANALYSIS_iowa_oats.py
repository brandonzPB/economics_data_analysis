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


iowa = read_data('iowa_full.csv')


# In[5]:


# setup linear regression for oat production
oat_reg = linear_model.LinearRegression()
oat_reg.fit(iowa[['Average Temp', 'Average Rain']], iowa['Oats'])

# coefficients
oat_reg.coef_


# In[6]:


# intercept
oat_reg.intercept_


# In[7]:


# get correlation between temperature and oat production
iowa['Average Temp'].corr(iowa['Oats'])


# In[8]:


# get correlation between rain and oat production
iowa['Average Rain'].corr(iowa['Oats'])


# In[9]:


# plot graph of oat production vs average temperature
sns.set_theme()

oat_temp_graph = sns.lmplot(
    data=iowa,
    x="Average Temp", y="Oats",
    height=5
)

oat_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Oat Production (1000 MT)")


# In[10]:


# plot graph of oat production vs average rainfall
oat_rain_graph = sns.lmplot(
    data=iowa,
    x="Average Rain", y="Oats",
    height=5
)

oat_rain_graph.set_axis_labels("Average Rainfall (inches)", "Oat Production (1000 MT)")


# In[ ]:




