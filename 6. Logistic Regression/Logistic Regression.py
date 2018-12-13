#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('insurance_data.csv')
df.head()


# In[6]:


plt.scatter(df.age, df.bought_insurance, marker='+', color ='red')


# In[7]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test,y_train, y_test =train_test_split(df[['age']], df.bought_insurance, train_size=0.9, test_size=0.1)


# In[11]:


X_test


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[18]:


model = LogisticRegression()


# In[19]:


model.fit(X_train, y_train)


# In[20]:


model.predict(X_test)


# In[21]:


X_test


# In[22]:


model.score(X_test,y_test)


# In[23]:


model.predict_proba(X_test) # specific to test case, to buy or not to buy


# In[ ]:




