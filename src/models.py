import numpy as np
import sklearn
import sklearn as skl
import pandas as pd
from graphs import *

def evaluation(y_test, y_pred):

    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    mape = sklearn.metrics.mean_absolute_percentage_error(y_test, y_pred)

    return mse, rmse, mae, mape

def linear_regression(x_train, y_train, x_test, y_test):
    #finding NaN values, getting ValueError for x_train
    #print("x_train:\n", x_train.isna().sum())
    #finding other possible rows that are NaN
    #print("y_train:\n", y_train.isna().sum())
    #print("x_test:\n", x_test.isna().sum())
    #print("y_test:\n", y_test.isna().sum())
    reg = skl.linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    #predicting load based on test features
    y_pred = reg.predict(x_test)

    mse, rmse, mae, mape = evaluation(y_test, y_pred)

    #print("MSE =", mse)
    #print("RMSE =", rmse)
    #print("MAE =", mae)
    #print(f"MAPE = {mape * 100}%")

    return y_pred
