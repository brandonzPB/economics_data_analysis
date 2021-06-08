#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn import linear_model


# In[2]:


def read_data(file):
    return pd.read_csv(file)


# In[5]:


miami = read_data('miami_full.csv')


# In[7]:


# setup linear regression for tangerine production
tang_reg = linear_model.LinearRegression()
tang_reg.fit(miami[['Average Temperature', 'Average Rainfall']], miami['Tangerines'])

# coefficients
tang_reg.coef_


# In[8]:


# intercept
tang_reg.intercept_


# In[9]:


# get correlation between temperature and tangerines produced
miami['Average Temperature'].corr(miami['Tangerines'])


# In[10]:


# get correlation between rainfall and tangerines produced
miami['Average Rainfall'].corr(miami['Tangerines'])


# In[11]:


# plot graph of tangerine production vs average temperature
sns.set_theme()

tang_temp_graph = sns.lmplot(
    data=miami,
    x="Average Temperature", y="Tangerines",
    height=5
)

tang_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Tangerines Produced (1000 MT)")


# In[ ]:





# In[12]:


# plot graph of tangerine production vs average rainfall
tang_rain_graph = sns.lmplot(
    data=miami,
    x="Average Rainfall", y="Tangerines",
    height=5
)

tang_rain_graph.set_axis_labels("Average Rainfall (inches)", "Tangerines Produced (1000 MT)")

