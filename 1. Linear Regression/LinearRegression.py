#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[42]:


df = pd.read_csv('homeprices.csv')


# In[43]:


df


# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price, color='red', marker='+')


# In[46]:


reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)


# In[59]:


reg.predict([[3500]])


# In[48]:


reg.coef_


# In[49]:


reg.intercept_


# In[50]:


d = pd.read_csv('areas.csv')


# In[51]:


d.head()


# In[52]:


p = reg.predict(d)


# In[53]:


p


# In[54]:


d['prices'] = p


# In[55]:


d


# In[58]:


d.to_csv('prediction.csv', index=False)


# In[ ]:




