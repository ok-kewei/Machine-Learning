#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
 


# In[3]:


df = pd.read_csv('homeprices_m.csv')


# In[14]:


df


# In[10]:


df.bedrooms.median()


# In[15]:


median_bedrooms = df.bedrooms.median()


# In[18]:


df.bedrooms = df.bedrooms.fillna(median_bedrooms)


# In[24]:


df


# In[27]:


reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)


# In[28]:


reg.coef_


# In[29]:


reg.intercept_


# In[30]:


reg.predict([[3000, 3, 40]])


# In[ ]:




