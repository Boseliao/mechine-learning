```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

#import plotly_express as px

from scipy import stats
from scipy.stats import norm, skew 

#import pandas_profiling

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
```

**Load Data**


```python
# Train data
df_train = pd.read_csv('./data/train.csv')
print("\nTrain data length:",df_train.shape)
print("\nTrain data columns:",df_train.columns)
print("\nTrain data info:",df_train.info())
print("\nTrain data:\n\n",df_train.head())
```

    
    Train data length: (1460, 81)
    
    Train data columns: Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition', 'SalePrice'],
          dtype='object')
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    
    Train data info: None
    
    Train data:
    
        Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
    0   1          60       RL         65.0     8450   Pave   NaN      Reg   
    1   2          20       RL         80.0     9600   Pave   NaN      Reg   
    2   3          60       RL         68.0    11250   Pave   NaN      IR1   
    3   4          70       RL         60.0     9550   Pave   NaN      IR1   
    4   5          60       RL         84.0    14260   Pave   NaN      IR1   
    
      LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold  \
    0         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   
    1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5   
    2         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9   
    3         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   
    4         Lvl    AllPub  ...        0    NaN   NaN         NaN       0     12   
    
      YrSold  SaleType  SaleCondition  SalePrice  
    0   2008        WD         Normal     208500  
    1   2007        WD         Normal     181500  
    2   2008        WD         Normal     223500  
    3   2006        WD        Abnorml     140000  
    4   2008        WD         Normal     250000  
    
    [5 rows x 81 columns]
    


```python
# Test data
df_test = pd.read_csv('./data/test.csv')
print("\nTest data length:",df_test.shape)
```

    
    Test data length: (1459, 80)
    

# **EDA**

- Do and same visualize


```python
sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# Plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='upper right')

ax = plt.axes()
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
```

    
     mu = 180921.20 and sigma = 79415.29
    
    

    D:\conda\lib\site-packages\ipykernel_launcher.py:11: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
      # This is added back by InteractiveShellApp.init_path()
    




    Text(0.5, 1.0, 'SalePrice distribution')




![png](output_6_3.png)



```python
# Let's Explore how SalePrice is distributed against normal theoretical quantiles
fig = plt.figure()
ax = fig.add_subplot()
res = stats.probplot(df_train['SalePrice'], plot=plt)
```


![png](output_7_0.png)


- build a correlation matrix


```python
# Correlation
df_train_corr = df_train.corr()
```


```python
df_train_corr[['SalePrice']].sort_values(by='SalePrice',ascending=False).style.background_gradient(cmap='viridis', axis=None)
```




<style  type="text/css" >
    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row0_col0 {
            background-color:  #fde725;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row1_col0 {
            background-color:  #84d44b;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row2_col0 {
            background-color:  #5ac864;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row3_col0 {
            background-color:  #3bbb75;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row4_col0 {
            background-color:  #37b878;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row5_col0 {
            background-color:  #32b67a;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row6_col0 {
            background-color:  #31b57b;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row7_col0 {
            background-color:  #25ab82;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row8_col0 {
            background-color:  #21a585;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row9_col0 {
            background-color:  #20a386;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row10_col0 {
            background-color:  #1fa088;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row11_col0 {
            background-color:  #1e9c89;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row12_col0 {
            background-color:  #1f9a8a;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row13_col0 {
            background-color:  #1f978b;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row14_col0 {
            background-color:  #24868e;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row15_col0 {
            background-color:  #277f8e;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row16_col0 {
            background-color:  #29798e;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row17_col0 {
            background-color:  #2a788e;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row18_col0 {
            background-color:  #2a778e;
            color:  #000000;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row19_col0 {
            background-color:  #2d718e;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row20_col0 {
            background-color:  #2e6d8e;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row21_col0 {
            background-color:  #32648e;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row22_col0 {
            background-color:  #34618d;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row23_col0 {
            background-color:  #39568c;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row24_col0 {
            background-color:  #3f4889;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row25_col0 {
            background-color:  #414487;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row26_col0 {
            background-color:  #453882;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row27_col0 {
            background-color:  #453781;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row28_col0 {
            background-color:  #482878;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row29_col0 {
            background-color:  #482576;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row30_col0 {
            background-color:  #482475;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row31_col0 {
            background-color:  #482475;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row32_col0 {
            background-color:  #482374;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row33_col0 {
            background-color:  #482374;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row34_col0 {
            background-color:  #481467;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row35_col0 {
            background-color:  #471164;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row36_col0 {
            background-color:  #440256;
            color:  #f1f1f1;
        }    #T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row37_col0 {
            background-color:  #440154;
            color:  #f1f1f1;
        }</style><table id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >SalePrice</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row0" class="row_heading level0 row0" >SalePrice</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row0_col0" class="data row0 col0" >1.000000</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row1" class="row_heading level0 row1" >OverallQual</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row1_col0" class="data row1 col0" >0.790982</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row2" class="row_heading level0 row2" >GrLivArea</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row2_col0" class="data row2 col0" >0.708624</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row3" class="row_heading level0 row3" >GarageCars</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row3_col0" class="data row3 col0" >0.640409</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row4" class="row_heading level0 row4" >GarageArea</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row4_col0" class="data row4 col0" >0.623431</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row5" class="row_heading level0 row5" >TotalBsmtSF</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row5_col0" class="data row5 col0" >0.613581</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row6" class="row_heading level0 row6" >1stFlrSF</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row6_col0" class="data row6 col0" >0.605852</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row7" class="row_heading level0 row7" >FullBath</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row7_col0" class="data row7 col0" >0.560664</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row8" class="row_heading level0 row8" >TotRmsAbvGrd</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row8_col0" class="data row8 col0" >0.533723</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row9" class="row_heading level0 row9" >YearBuilt</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row9_col0" class="data row9 col0" >0.522897</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row10" class="row_heading level0 row10" >YearRemodAdd</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row10_col0" class="data row10 col0" >0.507101</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row11" class="row_heading level0 row11" >GarageYrBlt</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row11_col0" class="data row11 col0" >0.486362</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row12" class="row_heading level0 row12" >MasVnrArea</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row12_col0" class="data row12 col0" >0.477493</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row13" class="row_heading level0 row13" >Fireplaces</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row13_col0" class="data row13 col0" >0.466929</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row14" class="row_heading level0 row14" >BsmtFinSF1</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row14_col0" class="data row14 col0" >0.386420</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row15" class="row_heading level0 row15" >LotFrontage</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row15_col0" class="data row15 col0" >0.351799</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row16" class="row_heading level0 row16" >WoodDeckSF</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row16_col0" class="data row16 col0" >0.324413</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row17" class="row_heading level0 row17" >2ndFlrSF</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row17_col0" class="data row17 col0" >0.319334</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row18" class="row_heading level0 row18" >OpenPorchSF</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row18_col0" class="data row18 col0" >0.315856</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row19" class="row_heading level0 row19" >HalfBath</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row19_col0" class="data row19 col0" >0.284108</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row20" class="row_heading level0 row20" >LotArea</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row20_col0" class="data row20 col0" >0.263843</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row21" class="row_heading level0 row21" >BsmtFullBath</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row21_col0" class="data row21 col0" >0.227122</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row22" class="row_heading level0 row22" >BsmtUnfSF</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row22_col0" class="data row22 col0" >0.214479</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row23" class="row_heading level0 row23" >BedroomAbvGr</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row23_col0" class="data row23 col0" >0.168213</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row24" class="row_heading level0 row24" >ScreenPorch</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row24_col0" class="data row24 col0" >0.111447</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row25" class="row_heading level0 row25" >PoolArea</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row25_col0" class="data row25 col0" >0.092404</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row26" class="row_heading level0 row26" >MoSold</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row26_col0" class="data row26 col0" >0.046432</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row27" class="row_heading level0 row27" >3SsnPorch</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row27_col0" class="data row27 col0" >0.044584</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row28" class="row_heading level0 row28" >BsmtFinSF2</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row28_col0" class="data row28 col0" >-0.011378</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row29" class="row_heading level0 row29" >BsmtHalfBath</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row29_col0" class="data row29 col0" >-0.016844</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row30" class="row_heading level0 row30" >MiscVal</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row30_col0" class="data row30 col0" >-0.021190</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row31" class="row_heading level0 row31" >Id</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row31_col0" class="data row31 col0" >-0.021917</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row32" class="row_heading level0 row32" >LowQualFinSF</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row32_col0" class="data row32 col0" >-0.025606</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row33" class="row_heading level0 row33" >YrSold</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row33_col0" class="data row33 col0" >-0.028923</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row34" class="row_heading level0 row34" >OverallCond</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row34_col0" class="data row34 col0" >-0.077856</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row35" class="row_heading level0 row35" >MSSubClass</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row35_col0" class="data row35 col0" >-0.084284</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row36" class="row_heading level0 row36" >EnclosedPorch</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row36_col0" class="data row36 col0" >-0.128578</td>
            </tr>
            <tr>
                        <th id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74level0_row37" class="row_heading level0 row37" >KitchenAbvGr</th>
                        <td id="T_739f4f52_8f7a_11eb_bbef_e86f38d08f74row37_col0" class="data row37 col0" >-0.135907</td>
            </tr>
    </tbody></table>



- explore the dataset for outliers


```python
# Scatter Plot
fig, ax = plt.subplots()
ax.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=12)
plt.xlabel('GrLivArea', fontsize=12)
plt.title('Sale Price', fontsize=16)
plt.show()
```


![png](output_12_0.png)


**异常点需要清洗**


```python
# Clean outliers
print("Length of data before dropping outliers:", len(df_train))
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) 
                                & (df_train['SalePrice']<300000)].index)
print("Length of data after dropping outliers:", len(df_train))
df_train = df_train.drop(df_train[(df_train['GrLivArea']>5000) 
                                | (df_train['SalePrice']>500000)].index)
print("Length of data after dropping outliers:", len(df_train))
```

    Length of data before dropping outliers: 1460
    Length of data after dropping outliers: 1458
    Length of data after dropping outliers: 1449
    

**箱型图处理异常点**


```python
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n
```

**数值型以及类别型数据**


```python
# Quantitative Variables
quan_var = [q for q in df_train.columns if df_train.dtypes[q] != 'object']
quan_var.remove('SalePrice') 
quan_var.remove('Id')
print("Quantitative Variables:\n", quan_var)

# Qualitative Variables
qual_var = [q for q in df_train.columns if df_train.dtypes[q] == 'object']
print("\nQualitative Variables:\n", qual_var)
```

    Quantitative Variables:
     ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    
    Qualitative Variables:
     ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
    

- Handling missing data


```python
# Calculate missing data ratio
df_train_na = (df_train.isnull().sum() / len(df_train)) * 100
df_train_na = df_train_na.drop(df_train_na[df_train_na == 0].index).sort_values(ascending=False)[:50]
missing_data = pd.DataFrame({'Missing Ratio' :df_train_na})
print('Missing data percentage:\n',missing_data.head(50))

# Plot
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
ax.set_facecolor("white")
sns.barplot(x=df_train_na.index, y=df_train_na)
sns.color_palette('pastel')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Percent of missing values', fontsize=12)
plt.title('Percent missing data by feature', fontsize=15)
```

    Missing data percentage:
                   Missing Ratio
    PoolQC            99.654934
    MiscFeature       96.273292
    Alley             93.719807
    Fence             80.676329
    FireplaceQu       47.619048
    LotFrontage       17.874396
    GarageYrBlt        5.590062
    GarageType         5.590062
    GarageFinish       5.590062
    GarageQual         5.590062
    GarageCond         5.590062
    BsmtFinType2       2.622498
    BsmtExposure       2.622498
    BsmtFinType1       2.553485
    BsmtCond           2.553485
    BsmtQual           2.553485
    MasVnrArea         0.552105
    MasVnrType         0.552105
    Electrical         0.069013
    




    Text(0.5, 1.0, 'Percent missing data by feature')




![png](output_20_2.png)


**为方便先合并训练集以及测试集**


```python
# Combine all data
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values
df_all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
df_all_data.drop(['Id','SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(df_all_data.shape))
```

    all_data size is : (2908, 79)
    

**扔掉丢失率大于70的列**


```python
# Get the list of variable based on missing data ratio
features_for_reg = missing_data[missing_data['Missing Ratio']>70].index.values.tolist()
df_all_data.drop(features_for_reg, axis=1, inplace=True)
```

**缺失值填充**


```python
features_for_reg = missing_data[missing_data['Missing Ratio']<70].index.values.tolist()

from sklearn.impute import SimpleImputer
imp1 = SimpleImputer(strategy='mean')
imp2 = SimpleImputer(strategy='most_frequent')
for col in features_for_reg:
    if col in quan_var:
        df_all_data[col] = imp1.fit_transform(df_all_data[[col]].values)
    elif col in qual_var:
        df_all_data[col] = imp2.fit_transform(df_all_data[[col]].values)
```

# 特征工程


```python
df_result = pd.DataFrame(columns=['Model','RMSE','MSE','Summary'])
print(df_result)
```

    Empty DataFrame
    Columns: [Model, RMSE, MSE, Summary]
    Index: []
    

- 类别特征构造


```python
X_all = pd.get_dummies(df_all_data)

X = X_all[0:len(df_train)]
y = df_train['SalePrice']

# Initiate train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(random_state=3)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Root Mean Squared Error: {:.2f}".format(rmse))

df_result = df_result.append(pd.DataFrame([['RandomForestRegressor'
                                            , rmse
                                            , mse
                                            ,'quan_var Features'                               
                                           ]], columns=df_result.columns))



# Calculate feature importances
importances = rf.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

print("Most important:\n", names[:10])
print("Least important:\n", names[(len(names)-10):])
```

    Root Mean Squared Error: 26211.78
    Most important:
     ['OverallQual_Garage_GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', 'OverallQual', 'GrLivArea', '1stFlrSF', 'LotArea', 'YearRemodAdd', 'YearBuilt', 'GarageArea']
    Least important:
     ['Condition2_Artery', 'Condition2_PosA', 'HeatingQC_Po', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'RoofStyle_Shed', 'RoofMatl_Membran', 'Exterior2nd_CBlock', 'RoofMatl_Roll']
    

- 特征选择


```python
# Get the list of variable based on rf feature importance
n_features = 150
features_for_reg = names[:n_features]


# Run Linear Regression
X_all = X_all[features_for_reg]

X = X_all[0:len(df_train)]
y = df_train['SalePrice']

# Initiate train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(random_state=3)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Root Mean Squared Error: {:.2f}".format(rmse))

df_result = df_result.append(pd.DataFrame([['RandomForestRegressor'
                                            , rmse
                                            , mse
                                            ,'Important features based on RF'                               
                                           ]], columns=df_result.columns))



# Calculate feature importances
importances = rf.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

print("Most important:\n", names[:10])
print("Least important:\n", names[(len(names)-10):])
```

    Root Mean Squared Error: 25918.07
    Most important:
     ['OverallQual_Garage_GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', 'OverallQual', 'GrLivArea', '1stFlrSF', 'LotArea', 'YearRemodAdd', 'YearBuilt', 'GarageArea']
    Least important:
     ['MasVnrType_BrkCmn', 'BldgType_TwnhsE', 'Condition1_Feedr', 'BsmtCond_Gd', 'BsmtQual_Fa', 'Neighborhood_NWAmes', 'Functional_Min2', 'RoofMatl_CompShg', 'Heating_GasW', 'LandSlope_Sev']
    


```python
# New feature
df_all_data["OverallQual_Garage_GrLivArea"] = df_all_data["OverallQual"] * df_all_data["GarageArea"] * df_all_data["GrLivArea"]

# Get Dummies
X_all = X_all.join(pd.get_dummies(df_all_data["OverallQual_Garage_GrLivArea"]))

X = X_all[0:len(df_train)]
y = df_train['SalePrice']

# Initiate train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(random_state=3)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Root Mean Squared Error: {:.2f}".format(rmse))
df_result = df_result.append(pd.DataFrame([['RandomForestRegressor'
                                            , rmse
                                            , mse
                                            ,'Features engineering'                               
                                           ]], columns=df_result.columns))



# Calculate feature importances
importances = rf.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

print("Most important:\n", names[:10])
print("Least important:\n", names[(len(names)-10):])
```

    Root Mean Squared Error: 26040.27
    Most important:
     ['OverallQual_Garage_GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', 'OverallQual', 'GrLivArea', '1stFlrSF', 'LotArea', 'YearRemodAdd', 'YearBuilt', 'GarageArea']
    Least important:
     [3567200.0, 3572370.0, 3577200.0, 3577824.0, 3579096.0, 7375872.0, 7369376.0, 3608064.0, 3610152.0, 58796300.0]
    


```python
# Get the list of variable based on missing data ratio
features_to_drop = names[(len(names)-10):]


# Get Dummies
X_all = pd.get_dummies(df_all_data[df_all_data.columns.difference(features_to_drop)])
X_all.fillna(0, inplace=True)

X = X_all[0:len(df_train)]
y = df_train['SalePrice']

# Initiate train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(random_state=3)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Root Mean Squared Error: {:.2f}".format(rmse))
df_result = df_result.append(pd.DataFrame([['RandomForestRegressor'
                                            , rmse
                                            , mse
                                            ,'Features Engineering and with less than 70% missing data'                               
                                           ]], columns=df_result.columns))



# Calculate feature importances
importances = rf.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

print("Most important:\n", names[:10])
print("Least important:\n", names[(len(names)-10):])
```

    Root Mean Squared Error: 25901.01
    Most important:
     ['OverallQual_Garage_GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', 'OverallQual', 'GrLivArea', '1stFlrSF', 'LotArea', 'YearRemodAdd', 'YearBuilt', 'GarageArea']
    Least important:
     ['RoofMatl_Roll', 'RoofStyle_Shed', 'RoofMatl_Membran', 'Condition1_RRNe', 'Condition2_RRAe', 'Condition2_RRAn', 'Electrical_Mix', 'ExterCond_Po', 'Exterior1st_CBlock', 'Heating_OthW']
    

# LGBM


```python
lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_lgb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
print("Root Mean Squared Error: {:.2f}".format(rmse))
df_result = df_result.append(pd.DataFrame([['LGBMRegressor'
                                            , rmse
                                            , mse
                                            ,'Features Engineering and with less than 70% missing data'                               
                                           ]], columns=df_result.columns))
```

    Root Mean Squared Error: 22941.50
    

**GridSearchCV网格搜索调参**


```python
lgb_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000, 2000, 5000],
               "max_depth": [2, 4, 5, 6, 8],
               "feature_fraction": [0.01, 0.02, 1],
               "colsample_bytree": [0.8, 1],
               'num_leaves': [2, 3, 4, 5, 6, 10]}
                              
lgb_cv_model = GridSearchCV(lgb_model,
                             lgb_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

print(lgb_cv_model.best_params_)
                              
lgb_tuned = LGBMRegressor(**lgb_cv_model.best_params_).fit(X_train, y_train)
y_pred_lgb = lgb_tuned.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lgb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
print("Root Mean Squared Error: {:.2f}".format(rmse))
df_result = df_result.append(pd.DataFrame([['LGBMRegressor'
                                            , rmse
                                            , mse
                                            ,'Tuned model with Features Engineering and with less than 70% missing data'                               
                                           ]], columns=df_result.columns))
```


```python

```
