#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('salaries.csv')
df.head()


# In[3]:


inputs = df.drop('salary_more_then_100k', axis ='columns')
target = df['salary_more_then_100k']


# In[4]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


leCompany = LabelEncoder()
leJob = LabelEncoder()
leDegree = LabelEncoder()


# In[7]:


inputs['company_n'] = leCompany.fit_transform(inputs['company'])
inputs['job_n'] = leJob.fit_transform(inputs['job'])
inputs['degree_n'] = leDegree.fit_transform(inputs['degree'])


# In[8]:


inputs['company_n']


# In[9]:


inputs.head()


# In[10]:


input_n = inputs.drop(['company', 'job', 'degree'], axis = 'columns')


# In[11]:


input_n


# In[12]:


from sklearn import tree


# In[14]:


model = tree.DecisionTreeClassifier()
model.fit(input_n, target)


# In[15]:


model.score(input_n,target) # score is 1.0 because we only use one set of data, that is no training test set.


# In[16]:


model.predict([[2,0,1]])


# In[ ]:




