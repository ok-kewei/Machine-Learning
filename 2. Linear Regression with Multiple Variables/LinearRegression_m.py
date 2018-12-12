#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('homeprices_m.csv')
df.bedrooms.median()
median_bedrooms = df.bedrooms.median()
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
reg.coef_
reg.intercept_

reg.predict([[3000, 3, 40]])

