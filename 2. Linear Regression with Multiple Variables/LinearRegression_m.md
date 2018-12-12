

```python
import pandas as pd
import numpy as np
from sklearn import linear_model
 
```


```python
df = pd.read_csv('homeprices_m.csv')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>bedrooms</th>
      <th>age</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>3.0</td>
      <td>20</td>
      <td>550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>4.0</td>
      <td>15</td>
      <td>565000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>NaN</td>
      <td>18</td>
      <td>610000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>3.0</td>
      <td>30</td>
      <td>595000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4000</td>
      <td>5.0</td>
      <td>8</td>
      <td>760000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4100</td>
      <td>6.0</td>
      <td>8</td>
      <td>810000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.bedrooms.median()
```




    4.0




```python
median_bedrooms = df.bedrooms.median()
```


```python
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>bedrooms</th>
      <th>age</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>3.0</td>
      <td>20</td>
      <td>550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>4.0</td>
      <td>15</td>
      <td>565000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>4.0</td>
      <td>18</td>
      <td>610000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>3.0</td>
      <td>30</td>
      <td>595000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4000</td>
      <td>5.0</td>
      <td>8</td>
      <td>760000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4100</td>
      <td>6.0</td>
      <td>8</td>
      <td>810000</td>
    </tr>
  </tbody>
</table>
</div>




```python
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
reg.coef_
```




    array([  112.06244194, 23388.88007794, -3231.71790863])




```python
reg.intercept_
```




    221323.00186540408




```python
reg.predict([[3000, 3, 40]])
```




    array([498408.25158031])




```python

```
