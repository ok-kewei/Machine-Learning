#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')

get_ipython().run_line_magic('matplotlib', 'inline')

plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price, color='red', marker='+')

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
reg.predict([[3500]])
reg.coef_
reg.intercept_

d = pd.read_csv('areas.csv')
d.head()
p = reg.predict(d)
d['prices'] = p


d.to_csv('prediction.csv', index=False)

