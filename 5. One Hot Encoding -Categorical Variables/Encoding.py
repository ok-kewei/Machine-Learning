#!/usr/bin/env python
# coding: utf-8

# In[209]:


import pandas as pd
df = pd.read_csv('homeprices.csv')

pd.get_dummies(df.town)
dummies = pd.get_dummies(df.town)

merged = pd.concat([df, dummies], axis='columns')
final = merged.drop(['town', 'west windsor'], axis = 'columns')

from sklearn.linear_model import LinearRegression
model = LinearRegression()

X = final.drop('price', axis = 'columns')
y = final.price
model.fit(X,y)
model.predict([[2800,0,1]]) #to predict for robinsville
model.predict([[2800,1,0]]) #to predict for monroe township
model.predict([[2800,0,0]]) #to predict for west windsor
model.score(X,y)


# In[225]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfle = df
le.fit_transform(dfle.town) # encode town to number
dfle.town =le.fit_transform(dfle.town) 

X = dfle[['town', 'area']].values
y = dfle.price

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

ohe=OneHotEncoder(categorical_features=[0])

X=ohe.fit_transform(X).toarray()
X = X[:,1:]

model.fit(X,y)
model.predict([[1,0,2800]])
model.predict([[0,1,3400]])

