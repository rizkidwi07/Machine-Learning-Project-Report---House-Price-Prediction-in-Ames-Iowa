# Machine Learning Project Report - House Price Prediction in Ames, Iowa
## Project Domain
The domain chosen for this machine learning project is Real Estate and Price Prediction, with the title Predictive Analytics: House Price Prediction in Ames, Iowa.

## Background
This project aims to predict the sale price of houses in Ames, Iowa, based on a series of property features. An accurate understanding of the factors influencing house prices is crucial for buyers, sellers, and investors in the real estate market. The dataset used is from Kaggle, "House Prices - Advanced Regression Techniques," which provides comprehensive data on residential properties in Ames, Iowa. By leveraging machine learning, we can develop a predictive model to assist in better decision-making for property transactions.

## Why and How This Problem Should Be Solved?
- Price Transparency: Many parties involved in property transactions (buyers, sellers, investors) often lack sufficient information regarding fair market prices. A price prediction model can bridge this information gap, providing objective and accurate price estimations.

- Better Decision-Making: Predictive models can help real estate agents and developers set competitive prices, while buyers can evaluate whether the offered price aligns with market value, thereby reducing the risk of overpaying or underselling.

- Market Efficiency: With more accurate price predictions, the real estate market becomes more efficient, fostering fairer transactions and reducing unhealthy speculation.

- Housing Accessibility: Predicting house prices can help people find properties that fit their budget, increasing accessibility to housing.

- Government Policy Support: Governments can use insights from prediction models to formulate more effective policies related to housing and urban development.

## Business Understanding
### Problem Statements
Based on the background above, here are the detailed problems that can be solved in this project:

- Among the available features, which ones are most influential in predicting house property prices?

- How can house prices be predicted using specific features?

### Goals
The objectives of this project are:

- To identify the features most correlated with house property prices in Ames, Iowa.

- To build a machine learning model that can predict house property prices in Ames, Iowa, as accurately as possible based on the available features.

### Solution Statements
- Methodology: Building a regression model with house property prices (SalePrice) as the target variable.

- Algorithms: Using machine learning algorithms such as K-Nearest Neighbor (KNN), Random Forest, and AdaBoost.

- Evaluation Metrics: Measuring model performance using Mean Squared Error (MSE) and accuracy score (R-squared).

## Data Understanding
The dataset used in this project is the "House Prices - Advanced Regression Techniques" dataset available on Kaggle. This dataset focuses on residential properties in Ames, Iowa, United States.

Dataset Information: 
Type | Keterangan
--- | ---
Title | House Prices - Advanced Regression Techniques
Source | [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
Maintainer | Dean De Cock (original creator)
License | Data files © Original Authors
Visibility | Public
Tags | Regression, Feature Engineering, Data Cleaning, EDA

## Variables in the Dataset
This dataset contains 81 columns with Id as an identifier and SalePrice as the target variable. Some key features relevant for price prediction based on square footage and number of bathrooms/bedrooms are:

- `LotFrontage`: Linear feet of street connected to property.
- `LotArea`: Lot size in square feet.
- `MasVnrArea`: Masonry veneer area in square feet.
- `BsmtFinSF1`: Type 1 finished square feet of basement area.
- `BsmtFinSF2`: Type 2 finished square feet of basement area (if present).
- `BsmtUnfSF`: Unfinished square feet of basement area.
- `TotalBsmtSF`: Total square feet of basement area.
- `1stFlrSF`: First Floor square feet.
- `2ndFlrSF`: Second floor square feet.
- `LowQualFinSF`: Low quality finished square feet (all floors).
- `GrLivArea`: Above grade (ground) living area square feet.
- `GarageArea`: Size of garage in square feet.
- `WoodDeckSF`: Wood deck area in square feet.
- `OpenPorchSF`: Open porch area in square feet.
- `EnclosedPorch`: Enclosed porch area in square feet.
- `3SsnPorch`: Three season porch area in square feet.
- `ScreenPorch`: Screen porch area in square feet.
- `PoolArea`: Pool area in square feet.
- `BsmtFullBath`: Basement full bathrooms.
- `BsmtHalfBath`: Basement half bathrooms.
- `FullBath`: Full bathrooms above grade.
- `HalfBath`: Half baths above grade.
- `BedroomAbvGr`: Number of bedrooms above basement level.
- `SalePrice`: The property's sale price in dollars. (Target Variable)

### Exploratory Data Analysis (EDA)
Univariate Analysis (Numerical Feature Distribution):
- `SalePrice`:

  - The `SalePrice` distribution is strongly right-skewed. This indicates that most houses have lower sale prices, with a few luxury properties having very high prices.

  - The majority of samples are concentrated at sale prices below approximately $300,000. As house prices increase, fewer properties are available in the dataset.

  - The price range varies widely, from around $34,900 to approximately $755,000.

  - This skewness has implications for regression models, potentially requiring data transformation (e.g., log transformation) for optimal performance.

- `Square Footage Features`:

  - Many features such as `LotArea`, `MasVnrArea`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, and `PoolArea` show strong right-skewed distributions or high concentrations at zero values. This indicates that the presence of these features is not universal across all properties. `PoolArea` is almost entirely zero, indicating the rarity of a swimming pool feature in this dataset.

  - `GrLivArea` (above grade living area) shows a more even distribution and has potential as an important predictor.

- Number of Bedrooms and Bathrooms Features:

  - BsmtFullBath and BsmtHalfBath are mostly 0 or 1, indicating basements rarely have multiple bathrooms.

  - FullBath and HalfBath show that most houses have 1 or 2 full bathrooms and 0 or 1 half bathrooms.

  - BedroomAbvGr has a fairly good distribution, with most houses having 2, 3, or 4 bedrooms, where 3 bedrooms is the most common.

- Multivariate Analysis (Relationships Between Numerical Features):
Strong Correlation with SalePrice:

  - GrLivArea (0.71): Shows a very strong positive correlation with SalePrice. The larger the above-ground living area, the higher the likely sale price. This is a highly dominant predictor.

  - GarageArea (0.62): Strong positive correlation. A larger garage area (or the presence of a garage) tends to correlate with a higher sale price.

  - TotalBsmtSF (0.61) and 1stFlrSF (0.61): Strong positive correlation. Larger total basement area and first floor area correlate with higher sale prices.

  - FullBath (0.56) and BedroomAbvGr (0.50): Strong positive correlation. More full bathrooms and bedrooms tend to lead to higher sale prices.

Moderate Correlation:

  - BsmtFinSF1 (0.37): Finished basement area type 1 has a moderate correlation.

  - LotFrontage (0.35) and LotArea (0.31): Lot frontage and lot area have a moderate relationship with sale price.

  - MasVnrArea (0.31) and OpenPorchSF (0.31): Masonry veneer area and open porch area show moderate correlations.

Weak/No Clear Pattern Correlation:

  - Features such as BsmtFinSF2 (-0.02), BsmtUnfSF (0.10), 2ndFlrSF (0.23), LowQualFinSF (-0.03), EnclosedPorch (-0.13), 3SsnPorch (0.04), ScreenPorch (0.10), PoolArea (0.09), BsmtFullBath (0.23), BsmtHalfBath (-0.01), and HalfBath (0.28) show very weak or near-zero correlation with SalePrice. Some even have very small negative correlations, indicating minimal contribution to price prediction.

Multicollinearity:

  - 1stFlrSF and TotalBsmtSF (0.82): Very high correlation.

  - GrLivArea and 1stFlrSF (0.81): Very high correlation.

  - High correlations between features representing area (e.g., GrLivArea, 1stFlrSF, TotalBsmtSF) indicate potential multicollinearity, which needs to be addressed during modeling to avoid issues with coefficient interpretation or model instability.

## Data Preparation
The techniques used in data preparation include:

- Handling Missing Values: Columns LotFrontage and MasVnrArea have missing values. Imputation strategies (filling in missing values) will be applied, such as using the median or mean, or more advanced imputation techniques if necessary.

- Handling Outliers: Outliers (extreme data points) will be detected using the IQR (InterQuartile Range) method and visualized with boxplots. These outliers will be handled to prevent negative impacts on model performance.

- Data Normalization: Numerical features will be normalized using StandardScaler to ensure a uniform data scale, which is crucial for optimal machine learning algorithm performance.

- Train-Test Split: The dataset will be split into training (90%) and testing (10%) sets to effectively build and evaluate the model.

## Modeling
In this modeling phase, three different regression algorithms will be implemented and evaluated:

- K-Nearest Neighbor (KNN) Regressor:

  - Uses sklearn.neighbors.KNeighborsRegressor.

  - Parameter used: n_neighbors = 10.

- Random Forest Regressor:

  - Uses sklearn.ensemble.RandomForestRegressor.

  - Parameters used: n_estimators = 50, max_depth = 16, random_state = 55, n_jobs = -1.

- AdaBoost Regressor:

  - Uses sklearn.ensemble.AdaBoostRegressor.

  - Parameters used: n_estimators = default, learning_rate = 0.05, random_state = 55.

## Evaluation
The main evaluation metrics used in this project are Mean Squared Error (MSE).

- Mean Squared Error (MSE): Measures the average of the squared differences between predicted values and actual values. The lower the MSE, the better the model's performance.

## Evaluation Results
**Model** | **MSE (Train)** | **MSE (Test)** 
--- | --- | ---
KNN | 6.221516e+08 | 1.340263e+09
Random Forest | **1.996160e+08** | **6.309028e+08**
AdaBoost | 1.144421e+09 |1.547844e+09

The Random Forest (RF) model demonstrates the best performance with the lowest MSE (both on training and test data). Meanwhile, the KNN model has the largest error. Therefore, Random Forest is selected as the best model for house price prediction.

## Impact of the Model on Business Understanding:
1. Does the model address the problem statements?

- Problem Statement 1: "Among the available features, which ones are most influential in predicting house property prices?"

  Impact: The Random Forest model intrinsically provides information regarding feature importance. Analysis from EDA and the correlation matrix indicates that features such as GrLivArea, TotalBsmtSF, 1stFlrSF, GarageArea, FullBath, and BedroomAbvGr have strong correlations with SalePrice and are likely dominant factors influencing property prices. This helps answer the question about the most influential features.

- Problem Statement 2: "How can house prices be predicted using specific features?"

  Impact: The Random Forest model generates accurate price predictions based on the features provided in the dataset. With this model, users (buyers, sellers, developers, or real estate agents) can estimate house prices based on property characteristics, enabling more informed decision-making.

2. Has the model successfully achieved the goals?

- Goal 1: "To identify the features most correlated with house property prices in Ames, Iowa."

  Achievement: The results of the EDA (histograms, pairplots, and correlation matrix) and the feature importance from Random Forest clearly identify the features most correlated with property prices, thus achieving this goal.

- Goal 2: "To build a machine learning model that can predict house property prices in Ames, Iowa, as accurately as possible based on the available features."

  Achievement: Through evaluation based on MSE and R-squared, the Random Forest model successfully achieved high prediction accuracy (R-squared 0.8872 on test data), demonstrating its ability to effectively predict house prices.

3. Is the planned solution statement impactful?

  - Impact: The planned methodology (using regression algorithms like KNN, Random Forest, and AdaBoost, with MSE as the primary metric) proved effective. The selection of Random Forest as the final model provides a good balance between accuracy and feature interpretability. In a business context, this solution not only answers analytical questions but also supports operational decisions, such as fair market pricing or sales strategies.

## Save Model
The best model, Random Forest, will be saved for future use without needing to retrain.
