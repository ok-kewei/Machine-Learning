#!/usr/bin/env python
# coding: utf-8

# In[209]:


import pandas as pd
df = pd.read_csv('homeprices.csv')
df


# In[210]:


pd.get_dummies(df.town)


# In[211]:


dummies = pd.get_dummies(df.town)


# In[212]:


dummies


# In[213]:


merged = pd.concat([df, dummies], axis='columns')


# In[214]:


merged


# In[215]:


final = merged.drop(['town', 'west windsor'], axis = 'columns')


# In[216]:


final


# In[217]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[218]:


X = final.drop('price', axis = 'columns')
X


# In[219]:


y = final.price
y


# In[220]:


model.fit(X,y)


# In[221]:


model.predict([[2800,0,1]]) #to predict for robinsville


# In[222]:


model.predict([[2800,1,0]]) #to predict for monroe township


# In[223]:


model.predict([[2800,0,0]]) #to predict for west windsor


# In[224]:


model.score(X,y)


# In[225]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[226]:


dfle = df
le.fit_transform(dfle.town) # encode town to number


# In[227]:


dfle.town =le.fit_transform(dfle.town) 


# In[228]:


dfle


# In[229]:


X = dfle[['town', 'area']].values
X


# In[230]:


y = dfle.price


# In[231]:


y


# In[232]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer


# In[233]:


ohe=OneHotEncoder(categorical_features=[0])


# In[234]:


X=ohe.fit_transform(X).toarray()


# In[235]:


X


# In[236]:


X = X[:,1:]


# In[237]:


X


# In[238]:


model.fit(X,y)


# In[240]:


model.predict([[1,0,2800]])


# In[242]:


model.predict([[0,1,3400]])


# In[ ]:




