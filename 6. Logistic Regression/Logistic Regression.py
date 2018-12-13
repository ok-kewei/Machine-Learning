#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv('insurance_data.csv')
df.head()

plt.scatter(df.age, df.bought_insurance, marker='+', color ='red')
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test =train_test_split(df[['age']], df.bought_insurance, train_size=0.9, test_size=0.1)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test,y_test)
model.predict_proba(X_test) # specific to test case, to buy or not to buy
