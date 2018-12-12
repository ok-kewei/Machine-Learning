#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
from sklearn import linear_model
 


# In[32]:


df = pd.read_csv('homeprices_m.csv')


# In[33]:


df


# In[34]:


df.bedrooms.median()


# In[35]:


median_bedrooms = df.bedrooms.median()


# In[36]:


df.bedrooms = df.bedrooms.fillna(median_bedrooms)


# In[37]:


df


# In[38]:


reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)


# In[39]:


reg.coef_


# In[40]:


reg.intercept_


# In[41]:


reg.predict([[3000, 3, 40]])


# In[44]:


import pickle


# In[45]:


with open('model_pickle','wb') as f:
    pickle.dump(reg,f)


# In[46]:


with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)


# In[50]:


mp.predict([[3000, 3, 40]])


# In[51]:


from sklearn.externals import joblib


# In[52]:


joblib.dump(reg, 'model_joblib')


# In[53]:


joblib.load('model_joblib')


# In[54]:


mk = joblib.load('model_joblib')


# In[55]:


mk.predict([[3000,3,40]])


# In[56]:


mk.coef_


# In[58]:


mp.coef_

