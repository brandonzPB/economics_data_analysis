#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import seaborn as sns
from sklearn import linear_model


# In[35]:


def read_data(file):
    return pd.read_csv(file)


# In[36]:


miami = read_data('miami_full.csv')


# In[37]:


# Get temperature average and variance

temp = miami['Average Temperature']

mean_temp = sum(temp) / len(temp)
print('Mean temp: ', mean_temp)

var_temp = sum((i - mean_temp) ** 2 for i in temp) / len(temp)
print('Variance in temp: ', var_temp)


# In[38]:


# Get rainfall average and variance

rain = miami['Average Rainfall']

mean_rain = sum(rain) / len(rain)
print('Mean rain: ', mean_rain)

var_rain = sum((i - mean_rain) ** 2 for i in rain) / len(rain)
print('Variance in rain: ', var_rain)


# In[39]:


# setup linear regression for orange production
orange_reg = linear_model.LinearRegression()
orange_reg.fit(miami[['Average Temperature', 'Average Rainfall']], miami['Oranges'])

# coefficients (oranges)
orange_reg.coef_


# In[40]:


# intercept value (oranges)
orange_reg.intercept_


# In[44]:


# Ensure accuracy of model:
# use average temperature and average rainfall...
# to confirm that prediction is average orange production
orange_reg.predict([[73.86, 5.23]])


# In[46]:


# get correlation between temperature and oranges produced
miami['Average Temperature'].corr(miami['Oranges'])


# In[47]:


# get correlation between rainfall and oranges produced
miami['Average Rainfall'].corr(miami['Oranges'])


# In[42]:


# plot graph of orange production vs average temperature
sns.set_theme()

orange_temp_graph = sns.lmplot(
    data=miami,
    x="Average Temperature", y="Oranges",
    height=5
)

orange_temp_graph.set_axis_labels("Average Temperature (Fahrenheit)", "Oranges Produced (1000 MT)")


# In[43]:


orange_rain_graph = sns.lmplot(
    data=miami,
    x="Average Rainfall", y="Oranges",
    height=5
)

orange_rain_graph.set_axis_labels("Average Rainfall (inches)", "Oranges Produced (1000 MT)")

