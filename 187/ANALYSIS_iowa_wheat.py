#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn import linear_model


# In[2]:


def read_data(file):
    return pd.read_csv(file)


# In[14]:


iowa = read_data('iowa_full.csv')


# In[15]:


# setup linear regression for wheat production
wheat_reg = linear_model.LinearRegression()
wheat_reg.fit(iowa[['Average Temp', 'Average Rain']], iowa['Wheat'])

# coefficients
wheat_reg.coef_


# In[16]:


# intercept
wheat_reg.intercept_


# In[17]:


# get correlation between temperature and wheat production
iowa['Average Temp'].corr(iowa['Wheat'])


# In[18]:


# get correlation between rain and wheat production
iowa['Average Rain'].corr(iowa['Wheat'])


# In[19]:


# plot grpah of wheat production vs average temperature
sns.set_theme()

wheat_temp_graph = sns.lmplot(
    data=iowa,
    x="Average Temp", y="Wheat",
    height=5
)

wheat_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Wheat Production (1000 MT)")


# In[20]:


# plot graph of wheat production vs average rain
wheat_rain_graph = sns.lmplot(
    data=iowa,
    x="Average Rain", y="Wheat",
    height=5
)

wheat_rain_graph.set_axis_labels("Average Rainfall (inches)", "Wheat Production (1000 MT)")


# In[ ]:




