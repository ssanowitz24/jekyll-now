
# Project 2


```python
#get tools
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
```


```python
#read in data
train = pd.read_csv("./project_2 data/train.csv", index_col='Id')
test = pd.read_csv("./project_2 data/test.csv", index_col ='Id')
```

### Distinguish Target and Features


```python
y_train = train['SalePrice']
features = [x for x in train.columns if x != 'SalePrice']
X_train = train[features]
for col in test.columns:
    if col not in X_train:
        X_train.drop(col, axis =1, inplace=True)
```

### Train Test Split


```python
Xtr, XT, ytr, yT = train_test_split(X_train, y_train, random_state = 24)

Xtr.isnull().sum()
```




    PID                  0
    MS SubClass          0
    MS Zoning            0
    Lot Frontage       239
    Lot Area             0
    Street               0
    Alley             1433
    Lot Shape            0
    Land Contour         0
    Utilities            0
    Lot Config           0
    Land Slope           0
    Neighborhood         0
    Condition 1          0
    Condition 2          0
    Bldg Type            0
    House Style          0
    Overall Qual         0
    Overall Cond         0
    Year Built           0
    Year Remod/Add       0
    Roof Style           0
    Roof Matl            0
    Exterior 1st         0
    Exterior 2nd         0
    Mas Vnr Type        13
    Mas Vnr Area        13
    Exter Qual           0
    Exter Cond           0
    Foundation           0
                      ... 
    Full Bath            0
    Half Bath            0
    Bedroom AbvGr        0
    Kitchen AbvGr        0
    Kitchen Qual         0
    TotRms AbvGrd        0
    Functional           0
    Fireplaces           0
    Fireplace Qu       752
    Garage Type         83
    Garage Yr Blt       84
    Garage Finish       84
    Garage Cars          1
    Garage Area          1
    Garage Qual         84
    Garage Cond         84
    Paved Drive          0
    Wood Deck SF         0
    Open Porch SF        0
    Enclosed Porch       0
    3Ssn Porch           0
    Screen Porch         0
    Pool Area            0
    Pool QC           1531
    Fence             1250
    Misc Feature      1491
    Misc Val             0
    Mo Sold              0
    Yr Sold              0
    Sale Type            0
    Length: 79, dtype: int64



### Munging


```python
Xtr.isnull().sum()[Xtr.isnull().sum() > 500]
```




    Alley           1433
    Fireplace Qu     752
    Pool QC         1531
    Fence           1250
    Misc Feature    1491
    dtype: int64




```python
#drop columns with more than 500 nulls
Xtra = Xtr.drop(['Alley', 'Fireplace Qu', 'Pool QC','Fence','Misc Feature'], axis=1)
```


```python
# replace null values with mean value of that column if less than or equal to 500 nulls
Xtra.sum()[(Xtra.isnull().sum() >= 1) & (Xtra.dtypes != 'object')]
  
for col in Xtra.isnull().sum()[(Xtra.isnull().sum() >= 1) & (Xtra.dtypes != 'object')].index:
    Xtra[col].fillna(Xtra[col].mean(), inplace =True)
   

```


```python
Xtra.isnull().sum()

```




    PID                 0
    MS SubClass         0
    MS Zoning           0
    Lot Frontage        0
    Lot Area            0
    Street              0
    Lot Shape           0
    Land Contour        0
    Utilities           0
    Lot Config          0
    Land Slope          0
    Neighborhood        0
    Condition 1         0
    Condition 2         0
    Bldg Type           0
    House Style         0
    Overall Qual        0
    Overall Cond        0
    Year Built          0
    Year Remod/Add      0
    Roof Style          0
    Roof Matl           0
    Exterior 1st        0
    Exterior 2nd        0
    Mas Vnr Type       13
    Mas Vnr Area        0
    Exter Qual          0
    Exter Cond          0
    Foundation          0
    Bsmt Qual          43
                       ..
    Low Qual Fin SF     0
    Gr Liv Area         0
    Bsmt Full Bath      0
    Bsmt Half Bath      0
    Full Bath           0
    Half Bath           0
    Bedroom AbvGr       0
    Kitchen AbvGr       0
    Kitchen Qual        0
    TotRms AbvGrd       0
    Functional          0
    Fireplaces          0
    Garage Type        83
    Garage Yr Blt       0
    Garage Finish      84
    Garage Cars         0
    Garage Area         0
    Garage Qual        84
    Garage Cond        84
    Paved Drive         0
    Wood Deck SF        0
    Open Porch SF       0
    Enclosed Porch      0
    3Ssn Porch          0
    Screen Porch        0
    Pool Area           0
    Misc Val            0
    Mo Sold             0
    Yr Sold             0
    Sale Type           0
    Length: 74, dtype: int64




```python
#Now make dummies on all objects
obj_features = [x for x in Xtra if Xtra[x].dtypes == 'object']
Xtra_obj = Xtra[obj_features]
Xtra_dum = pd.get_dummies(Xtra_obj)
```


```python
Xtra = pd.concat([Xtra, Xtra_dum], axis=1)
Xtra.head()
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
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>Utilities</th>
      <th>Lot Config</th>
      <th>...</th>
      <th>Paved Drive_Y</th>
      <th>Sale Type_COD</th>
      <th>Sale Type_CWD</th>
      <th>Sale Type_Con</th>
      <th>Sale Type_ConLD</th>
      <th>Sale Type_ConLI</th>
      <th>Sale Type_ConLw</th>
      <th>Sale Type_New</th>
      <th>Sale Type_Oth</th>
      <th>Sale Type_WD</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>324</th>
      <td>923202105</td>
      <td>20</td>
      <td>RL</td>
      <td>93.0</td>
      <td>10114</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1739</th>
      <td>528222060</td>
      <td>60</td>
      <td>RL</td>
      <td>72.0</td>
      <td>8229</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2835</th>
      <td>908227010</td>
      <td>50</td>
      <td>RL</td>
      <td>72.0</td>
      <td>7822</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Bnk</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>902404010</td>
      <td>30</td>
      <td>RM</td>
      <td>90.0</td>
      <td>8100</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>362</th>
      <td>527164090</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8125</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 303 columns</p>
</div>




```python
#Now that dummies are made get rid of object columns
Xtra_noobj = Xtra.select_dtypes(exclude = 'object')
Xtra_noobj.shape
```




    (1538, 266)




```python
Xtra_noobj.info() #nice
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1538 entries, 324 to 998
    Columns: 266 entries, PID to Sale Type_WD 
    dtypes: float64(11), int64(26), uint8(229)
    memory usage: 800.5 KB


### Clean the Test Set


```python
#follow same procedure as training set
XT.isnull().sum()[XT.isnull().sum() > 500]
XTa = XT.drop(['Alley', 'Fireplace Qu', 'Pool QC','Fence','Misc Feature'], axis=1)
```


```python

  
for col in XTa.isnull().sum()[(XTa.isnull().sum() >= 1) & (XTa.dtypes != 'object')].index:
    XTa[col].fillna(XTa[col].mean(), inplace =True)
   
```


```python
XTa.shape

```




    (513, 74)




```python
obj_features = [x for x in XTa if XTa[x].dtypes == 'object']
XTa_obj = XTa[obj_features]
XTa_dum = pd.get_dummies(XTa_obj)
```


```python
XTa = pd.concat([XTa, XTa_dum], axis=1)
XTa.head()
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
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>Utilities</th>
      <th>Lot Config</th>
      <th>...</th>
      <th>Garage Cond_TA</th>
      <th>Paved Drive_N</th>
      <th>Paved Drive_P</th>
      <th>Paved Drive_Y</th>
      <th>Sale Type_COD</th>
      <th>Sale Type_Con</th>
      <th>Sale Type_ConLD</th>
      <th>Sale Type_ConLI</th>
      <th>Sale Type_New</th>
      <th>Sale Type_WD</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>535</th>
      <td>531363030</td>
      <td>20</td>
      <td>RL</td>
      <td>63.0</td>
      <td>7500</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2815</th>
      <td>907420070</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8461</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>910</th>
      <td>909179020</td>
      <td>75</td>
      <td>RL</td>
      <td>102.0</td>
      <td>15863</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>527105010</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1824</th>
      <td>532376090</td>
      <td>20</td>
      <td>RL</td>
      <td>40.0</td>
      <td>13673</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 279 columns</p>
</div>




```python
XTa_noobj = XTa.select_dtypes(exclude = 'object')
XTa_noobj.shape
```




    (513, 242)




```python
#need same number of col
for col in Xtra_noobj:
    if col not in XTa_noobj:
        Xtra_noobj.drop(col, axis =1, inplace=True)
Xtra_noobj.shape
for col in XTa_noobj:
    if col not in Xtra_noobj:
        XTa_noobj.drop(col, axis =1, inplace=True)
XTa_noobj.shape
Xtra_noobj.shape
```

    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy





    (1538, 238)



### LASSO


```python
#run a LASSO model on training data, Scale, instantiate, fit, score
ss = StandardScaler()
ss_Xtr =ss.fit_transform(Xtra_noobj)
l_alphas = np.arange(0.001, 0.2, 0.0001) 
lasso_m = LassoCV(alphas = l_alphas, cv=3)
lasso_m =lasso_m.fit(ss_Xtr, ytr)
lasso_optimal_alpha = lasso_m.alpha_
print(lasso_optimal_alpha,':is the optimal alpha')

```

    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    0.1992000000000001 :is the optimal alpha


    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)



```python
ss_XT = ss.fit_transform(XTa_noobj)
print("R2 score:",lasso_m.score(ss_Xtr, ytr))
y_hat = lasso_m.predict(ss_XT)
sns.regplot(yT, y_hat)
```

    R2 score: 0.9138328394816564





    <matplotlib.axes._subplots.AxesSubplot at 0x1a0db16f98>




![png](/images/Cleaner_book_files/Cleaner_book_26_2.png)


### Prediction Data


```python
# try on real test data, prediction data
#follow same munging procedure as other data sets
test.drop(['Alley', 'Fireplace Qu', 'Pool QC','Fence','Misc Feature'], axis=1, inplace=True)
```


```python
for col in test.isnull().sum()[(test.isnull().sum() >= 1) & (test.dtypes != 'object')].index:
    test[col].fillna(test[col].mean(), inplace =True)
```


```python
obj_features = [x for x in test if test[x].dtypes == 'object']
test_obj = test[obj_features]
test_dum = pd.get_dummies(test_obj)
```


```python
test = pd.concat([test, test_dum], axis=1)
test.head()
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
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>Utilities</th>
      <th>Lot Config</th>
      <th>...</th>
      <th>Sale Type_COD</th>
      <th>Sale Type_CWD</th>
      <th>Sale Type_Con</th>
      <th>Sale Type_ConLD</th>
      <th>Sale Type_ConLI</th>
      <th>Sale Type_ConLw</th>
      <th>Sale Type_New</th>
      <th>Sale Type_Oth</th>
      <th>Sale Type_VWD</th>
      <th>Sale Type_WD</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2658</th>
      <td>902301120</td>
      <td>190</td>
      <td>RM</td>
      <td>69.000000</td>
      <td>9142</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2718</th>
      <td>905108090</td>
      <td>90</td>
      <td>RL</td>
      <td>69.630042</td>
      <td>9662</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2414</th>
      <td>528218130</td>
      <td>60</td>
      <td>RL</td>
      <td>58.000000</td>
      <td>17104</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>902207150</td>
      <td>30</td>
      <td>RM</td>
      <td>60.000000</td>
      <td>8520</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>625</th>
      <td>535105100</td>
      <td>20</td>
      <td>RL</td>
      <td>69.630042</td>
      <td>9500</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 293 columns</p>
</div>




```python
test_noobj = test.select_dtypes(exclude = 'object')
test_noobj.info()
test_noobj.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 879 entries, 2658 to 1939
    Columns: 256 entries, PID to Sale Type_WD 
    dtypes: float64(11), int64(26), uint8(219)
    memory usage: 448.9 KB





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
      <th>PID</th>
      <th>MS SubClass</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Mas Vnr Area</th>
      <th>BsmtFin SF 1</th>
      <th>...</th>
      <th>Sale Type_COD</th>
      <th>Sale Type_CWD</th>
      <th>Sale Type_Con</th>
      <th>Sale Type_ConLD</th>
      <th>Sale Type_ConLI</th>
      <th>Sale Type_ConLw</th>
      <th>Sale Type_New</th>
      <th>Sale Type_Oth</th>
      <th>Sale Type_VWD</th>
      <th>Sale Type_WD</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2658</th>
      <td>902301120</td>
      <td>190</td>
      <td>69.000000</td>
      <td>9142</td>
      <td>6</td>
      <td>8</td>
      <td>1910</td>
      <td>1950</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2718</th>
      <td>905108090</td>
      <td>90</td>
      <td>69.630042</td>
      <td>9662</td>
      <td>5</td>
      <td>4</td>
      <td>1977</td>
      <td>1977</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2414</th>
      <td>528218130</td>
      <td>60</td>
      <td>58.000000</td>
      <td>17104</td>
      <td>7</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>0.0</td>
      <td>554.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>902207150</td>
      <td>30</td>
      <td>60.000000</td>
      <td>8520</td>
      <td>5</td>
      <td>6</td>
      <td>1923</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>625</th>
      <td>535105100</td>
      <td>20</td>
      <td>69.630042</td>
      <td>9500</td>
      <td>6</td>
      <td>5</td>
      <td>1963</td>
      <td>1963</td>
      <td>247.0</td>
      <td>609.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 256 columns</p>
</div>




```python
feats = [col for col in Xtra_noobj if col in test_noobj]
testa = test_noobj[feats]
Xtra = Xtra_noobj[feats]
testa.shape
Xtra.shape
```




    (1538, 231)




```python
ss_test = ss.fit_transform(testa)
```

### LASSO model: let's predict and get a Kaggle Score


```python
ss_Xtr = ss.fit_transform(Xtra)
l_alphas = np.arange(0.001, 0.2, 0.0001) 
lasso_m = LassoCV(alphas = l_alphas, cv=3)
lasso_m =lasso_m.fit(ss_Xtr, ytr)
lasso_optimal_alpha = lasso_m.alpha_
print(lasso_optimal_alpha,':is the optimal alpha')

y_hat1 = lasso_m.predict(ss_test)

```

    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    0.19990000000000008 :is the optimal alpha


    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)



```python
#set up for Kaggle submission
a_trial = pd.DataFrame(y_hat1, index = testa.index)
a_trial.columns = ['SalePrice']
a_trial.head()

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
      <th>SalePrice</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2658</th>
      <td>245827.222904</td>
    </tr>
    <tr>
      <th>2718</th>
      <td>155142.898489</td>
    </tr>
    <tr>
      <th>2414</th>
      <td>214997.650073</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>118550.063245</td>
    </tr>
    <tr>
      <th>625</th>
      <td>189368.631778</td>
    </tr>
  </tbody>
</table>
</div>




```python
a_trial.to_csv("./project_2 data/a_trial.csv")
```

### Elastic Net Model with feature engineering via Polynomial Features


```python
# feature engineering
poly = PolynomialFeatures()
poly_Xtr = poly.fit_transform(Xtra_noobj)
poly_XT = poly.fit_transform(XTa_noobj)

#standardize
ss_poly_Xtr = ss.fit_transform(poly_Xtr)
ss_poly_XT = ss.fit_transform(poly_XT)





```

### Elastic Net Model on training data


```python
E_net_alphas = np.arange(0.5, 1, 0.05)
E_net_ratio = np.arange(0.5, 1, 0.05)
E_net_model = ElasticNetCV(alphas = E_net_alphas, l1_ratio=E_net_ratio)
E_net_model = E_net_model.fit(ss_poly_Xtr, ytr)
E_net_optimal_alpha = E_net_model.alpha_
print(E_net_model.l1_ratio_,"is the optimal l1_ratio")
print(E_net_optimal_alpha,"is the optimal alpha")
```

    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    0.5 is the optimal l1_ratio
    0.9000000000000004 is the optimal alpha


    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)



```python
print(E_net_model.score(ss_poly_Xtr, ytr))
y_hatE =E_net_model.predict(ss_poly_XT)

sns.regplot(yT, y_hatE)
```

    0.9943844897247845





    <matplotlib.axes._subplots.AxesSubplot at 0x1a0c9f15f8>




![png](/images/Cleaner_book_files/Cleaner_book_43_2.png)


### Elastic Net on Prediction Data


```python
feats = [col for col in Xtra_noobj if col in test_noobj]
testb = test_noobj[feats]
Xtrb = Xtra_noobj[feats]
testb.shape
Xtrb.shape
```




    (1538, 231)




```python
# feature engineering
poly = PolynomialFeatures()
poly_Xtrb = poly.fit_transform(Xtrb)
poly_testb = poly.fit_transform(testb)

#standardize
ss_poly_Xtrb = ss.fit_transform(poly_Xtrb)
ss_poly_testb = ss.fit_transform(poly_testb)
```


```python
poly_Xtrb.shape
```




    (1538, 27028)




```python
E_net_alphas = np.arange(0.5, 1, 0.05)
E_net_ratio = np.arange(0.5, 1, 0.05)
E_net_model = ElasticNetCV(alphas = E_net_alphas, l1_ratio=E_net_ratio)
E_net_model = E_net_model.fit(ss_poly_Xtrb, ytr)
E_net_optimal_alpha = E_net_model.alpha_
print(E_net_model.l1_ratio_,"is the optimal l1_ratio")
print(E_net_optimal_alpha,"is the optimal alpha")
```

    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    0.5 is the optimal l1_ratio
    0.9000000000000004 is the optimal alpha


    /Users/scottsanowitz/anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)



```python
#prediction with Elastic Net
y_hatb = E_net_model.predict(ss_poly_testb)
```


```python
#set up for kaggle
b_trial = pd.DataFrame(y_hatb, index = testb.index)
b_trial.columns = ['SalePrice']
b_trial.head()
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
      <th>SalePrice</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2658</th>
      <td>163776.457176</td>
    </tr>
    <tr>
      <th>2718</th>
      <td>160671.749433</td>
    </tr>
    <tr>
      <th>2414</th>
      <td>198968.742629</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>113674.435154</td>
    </tr>
    <tr>
      <th>625</th>
      <td>183432.029082</td>
    </tr>
  </tbody>
</table>
</div>




```python
b_trial.to_csv("./project_2 data/b_trial.csv")
```
