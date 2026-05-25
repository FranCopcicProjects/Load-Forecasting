import numpy as np
import sklearn
import sklearn as skl
import pandas as pd
import xgboost as xgb
from flaml import AutoML
from data_manager import *
import matplotlib.pyplot as plt
import tensorflow.keras as keras

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
    print("LINEAR REGRESSION:")
    print("MSE =", mse)
    print("RMSE =", rmse)
    print("MAE =", mae)
    print(f"MAPE = {mape * 100}%")
    print()

    return y_pred

def random_forest(x_train, y_train, x_test, y_test):
    reg = skl.ensemble.RandomForestRegressor()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    mse, rmse, mae, mape = evaluation(y_test, y_pred)
    print("RANDOM FOREST:")
    print("MSE =", mse)
    print("RMSE =", rmse)
    print("MAE =", mae)
    print(f"MAPE = {mape * 100}%")
    print()

    return y_pred

def xgboost(x_train, y_train, x_test, y_test):
    reg = xgb.XGBRegressor()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    mse, rmse, mae, mape = evaluation(y_test, y_pred)
    print("XGBOOST:")
    print("MSE =", mse)
    print("RMSE =", rmse)
    print("MAE =", mae)
    print(f"MAPE = {mape * 100}%")
    print()

    return y_pred

def automl(x_train, y_train, x_test, y_test):
    reg = AutoML()
    reg.fit(X_train=x_train, y_train=y_train, task="regression", time_budget=60)
    y_pred = reg.predict(x_test)
    mse, rmse, mae, mape = evaluation(y_test, y_pred)
    print("AUTOML:")
    print("Best model =", reg.best_estimator)
    print("MSE =", mse)
    print("RMSE =", rmse)
    print("MAE =", mae)
    print(f"MAPE = {mape * 100}%")
    print()
    return y_pred

def train_neural_network(X, y, WINDOW_SIZE, epochs=10, learning_rate=0.005):
    model = keras.models.Sequential()
    num_features = X.shape[2]
    model.add(keras.layers.InputLayer(input_shape=(WINDOW_SIZE, num_features)))
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dense(8, activation="relu"))
    model.add(keras.layers.Dense(1, activation="linear"))

    check_point = keras.callbacks.ModelCheckpoint("../data/model.keras", monitor="loss", save_best_only=True)
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=[keras.metrics.RootMeanSquaredError()])

    model.fit(X, y, epochs=epochs, callbacks=[check_point])
    return model

def lstm(train, test, WINDOW_SIZE):
    train_input, train_output = train
    test_input, test_output = test

    print(train_input.shape, train_output.shape)
    print(test_input.shape, test_output.shape)


    model = train_neural_network(train_input, train_output, WINDOW_SIZE)

    y_pred = model.predict(test_input)
    y_pred = y_pred.flatten()
    mse, rmse, mae, mape = evaluation(test_output, y_pred)

    print("LSTM:")
    print("MSE =", mse)
    print("RMSE =", rmse)
    print("MAE =", mae)
    print(f"MAPE = {mape * 100}%")
    print()

    return y_pred