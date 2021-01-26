import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer # for encoding categorical to numerical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # for splitting data into trains and tests
from sklearn.linear_model import LinearRegression # for training and predicting
from sklearn.preprocessing import PolynomialFeatures # for polynomial linear regression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor # for random forest regression

def RFR(): # the code is similar to decision tree
    # import data
    data  = pd.read_csv('Position_Salaries.csv')

    # independent variable
    x = data.iloc[:, 1:-1].values

    # dependent variable
    y = data.iloc[:, -1].values
    # print(x, y)

    # splitting into 4 parts, but it is too small
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

    # training dataset
    rfr_reg = RandomForestRegressor(n_estimators=1000, random_state=1) # number of trees (default is 100)
    rfr_reg.fit(x, y)

    # feature scaling


    # prediction
    print('6.5 years Salary will be %d dollars' %rfr_reg.predict([[6.5]]))

    # Visualization the results with smoothness
    # Decision Tree should follow this side
    x_grid = np.arange(min(x), max(x), 0.1)
    x_grid = x_grid.reshape(len(x_grid), 1)
    plt.scatter(x, y, color='red', marker='X')
    plt.plot(x_grid, rfr_reg.predict(x_grid), color='blue')
    plt.title('Random Forest Regression')
    plt.xlabel('Position')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    RFR()
