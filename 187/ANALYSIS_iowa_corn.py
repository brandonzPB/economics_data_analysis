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


iowa = read_data('iowa_full.csv')


# In[5]:


# setup linear regression for corn production
corn_reg = linear_model.LinearRegression()
corn_reg.fit(iowa[['Average Temp', 'Average Rain']], iowa['Corn'])

# coefficients
corn_reg.coef_


# In[6]:


# intercept
corn_reg.intercept_


# In[7]:


# get correlation between temperature and corn production
iowa['Average Temp'].corr(iowa['Corn'])


# In[8]:


# get correlation between rainfall and corn production
iowa['Average Rain'].corr(iowa['Corn'])


# In[9]:


# plot graph of corn production vs average temperature
sns.set_theme()

corn_temp_graph = sns.lmplot(
    data=iowa,
    x="Average Temp", y="Corn",
    height=5
)

corn_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Corn Production (1000 MT)")


# In[10]:


# plot graph of corn production vs average rainfall
corn_rain_graph = sns.lmplot(
    data=iowa,
    x="Average Rain", y="Corn",
    height=5
)

corn_rain_graph.set_axis_labels("Average Rainfall (inches)", "Corn Production (1000 MT)")


# In[ ]:




