# -*- coding: utf-8 -*-
"""Predict House Prices - Linear Regression Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1a9Vm9gSujjGEW5KgKoIA5GyP1SBk4KHe

#Data Loading#

##Import Library
"""

# Commented out IPython magic to ensure Python compatibility.
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

"""## Import Dataset"""

from google.colab import files
files.upload()

# Check dataset
house = pd.read_csv('/content/train.csv')
house

"""* There are 1.460 rows in the dataset.
* There are 81 columns in the dataset.
* In this case, a linear regression model will be applied to predict house prices based on floor area and the number of bedrooms and bathrooms.
* Based on the columns above, the following are columns related to square footage and the number of bedrooms and bathrooms:
`LotFrontage`, `LotArea`, `MasVnrArea`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `GrLivArea`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch` `ScreenPorch`, `PoolArea`, `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`, `BedroomAbvGr`, and `SalePrice` as a target variable.
"""

# display the desired columns
desired_column = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'SalePrice'
]

# select only the desired columns
house_option = house[desired_column]

house_option

"""Now there are only 24 columns

# Exploratory Data Analysis (EDA)

The variables in the House Prices - Advanced Regression Techniques are as follows:
* `LotFrontage`: Linear feet of street connected to property
* `LotArea`: Lot size in square feet
* `MasVnrArea`: Masonry veneer type
* `BsmtFinSF1`: Type 1 finished square feet
* `BsmtFinSF2`: Type 2 finished square feet
* `BsmtUnfSF`: Unfinished square feet of basement area
* `TotalBsmtSF`: Total square feet of basement area
* `1stFlrSF`: First Floor square feet
* `2ndFlrSF`: Second floor square feet
* `LowQualFinSF`: Low quality finished square feet (all floors)
* `GrLivArea`: Above grade (ground) living area square feet
* `GarageArea`: Size of garage in square feet
* `WoodDeckSF`: Wood deck area in square feet
* `OpenPorchSF`: Open porch area in square feet
* `EnclosedPorch`: Enclosed porch area in square feet
* `3SsnPorch`: Three season porch area in square feet
* `ScreenPorch`: Screen porch area in square feet
* `PoolArea`: Pool area in square feet
* `BsmtFullBath`: Basement full bathrooms
* `BsmtHalfBath`: Basement half bathrooms
* `FullBath`: Full bathrooms above grade
* `HalfBath`: Half baths above grade
* `Bedroom`: Number of bedrooms above basement level
* `SalePrice`: The property's sale price in dollars
"""

house_option.info()

"""From the output, it can be seen that:

* The dataset consists of 1.460 rows and 24 columns.
* There are missing values in several columns, including `LotFrontage` and `MasVnrArea`
"""

house_option.describe()

"""From the results of the describe() function, the minimum values for the several columns are 0. This indicates that there are some features that are not present in the house.

## Univariate Analysis
"""

# analysis of the number of unique values in each numerical feature
house_option.hist(bins=50, figsize=(20,15))
plt.show()

"""* Many square footage and exterior area features show a strong right-skewed distribution or high concentration at zero values, indicating that the presence of these features is not universal across all properties. This may require special handling.
* Features such as `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `LotArea`, and the number of bedrooms/bathrooms (`FullBath`, `BedroomAbvGr`) exhibit more substantial variation and tend to be important predictors in house price models.
* The right-skewed `SalePrice` is a common finding in property price data and indicates the need for special attention during the data preprocessing stage for regression models.

## Multivariate Analysis
"""

# analysis of the relationship between numerical features
sns.pairplot(house_option, diag_kind = 'kde')

"""This pairplot clearly highlights that the above-ground living area (`GrLivArea`), total basement area (`TotalBsmtSF`), first floor area (`1stFlrSF`), garage area (`GarageArea`), and number of full bathrooms (`FullBath`) are very strong predictors and positively correlated with SalePrice. These features are likely to be key components in your regression model. On the other hand, many other features, especially those related to additional areas that are rare or of low quality, show very weak correlations with `SalePrice`, indicating that they may be less informative for predicting sale prices. To evaluate the correlation score, use the corr() function."""

plt.figure(figsize=(15, 10))
correlation_matrix = house_option.corr().round(2)

# To print the value inside the box, use the parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.8, )
plt.title("Correlation Matrix ", size=50)

"""* Features with very strong correlations (`GrLivArea`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `BedroomAbvGr`) are prime candidates for use in the model because they have a clear relationship with `SalePrice`.
* Features with high multicollinearity (e.g., `1stFlrSF` and `GrLivArea`, or `1stFlrSF` and `TotalBsmtSF`) require attention. Including both features with very high correlations in a linear regression model simultaneously can cause multicollinearity issues, which may make interpreting coefficients difficult or reduce model stability. It may need to choose one or combine them.
* Features with very weak or near-zero correlations with `SalePrice` (`BsmtFinSF2`, `LowQualFinSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, `PoolArea`, `BsmtHalfBath`) are unlikely to contribute significantly to model performance and may be considered for removal to simplify the model and reduce noise.

# Data Preparation

## Delete Irrelevant Columns
"""

# display the desired columns
desired_column = [
    'GrLivArea', 'GarageArea', 'TotalBsmtSF',
    '1stFlrSF', 'FullBath', 'BedroomAbvGr', 'SalePrice'
]

# select only the desired columns
house_option = house[desired_column]

house_option

"""## Handling Missing Values"""

print("Total missing value: ", house_option.isna().sum())

"""From the output above, there are no rows with missing values."""

house_option.info()

"""## Handling Duplication Issues"""

print("Total duplication: ", house_option.duplicated().sum())

house_option.drop_duplicates(inplace=True)

house_option.info()

"""## Handling Outliers"""

sns.boxplot(x=house_option['GrLivArea'])

sns.boxplot(x=house_option['GarageArea'])

sns.boxplot(x=house_option['TotalBsmtSF'])

sns.boxplot(x=house_option['1stFlrSF'])

sns.boxplot(x=house_option['FullBath'])

sns.boxplot(x=house_option['BedroomAbvGr'])

sns.boxplot(x=house_option['SalePrice'])

Q1 = house_option.quantile(0.25)
Q3 = house_option.quantile(0.75)
IQR = Q3 - Q1
house_option = house_option[~((house_option < (Q1 - 1.5 * IQR)) | (house_option > (Q3 + 1.5 * IQR))).any(axis=1)]

house_option.info()

"""Now, the data has been cleaned.

## Train-Test-Split
"""

from sklearn.model_selection import train_test_split

X = house_option.drop(["SalePrice"],axis =1)
y = house_option["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""## Normalization"""

from sklearn.preprocessing import StandardScaler

numerical_features = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'BedroomAbvGr']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

"""Now the mean value = 0 and the standard deviation = 1

# Modelling
"""

# Prepare the dataframe for model analysis
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""## K-Nearest Neighbour"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""## Random Forest"""

from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""## AdaBoost"""

from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""# Evaluation"""

# Scale the numerical features in X_test so that they have a mean of 0 and a variance of 1.
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Create a variable mse containing the mse values for the train and test data for each algorithm.
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

# Create a dictionary for each algorithm used.
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Calculate the Mean Squared Error for each algorithm on the train and test data
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

# Call mse
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""The Random Forest (RF) model provides the smallest error value. Meanwhile, the model with the Boosting algorithm has the largest error. Therefore, we will choose the Random Forest (RF) model as the best model for predicting house prices."""

prediction = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediction_'+name] = model.predict(prediction).round(1)

pd.DataFrame(pred_dict)

"""It appears that predictions using Random Forest (RF) provide the closest results.

# Save Model
"""

import joblib

# Save the Random Forest model since it performed the best
joblib.dump(RF, 'houseprice_model.joblib')

print("Model saved as houseprice_model.joblib")