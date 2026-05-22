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

def linear_regression(x_train, y_train, time_train, x_test, y_test, time_test, graph_type = "daily", start_date_input = "2025-01-01"):
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

    #get graphs
    if graph_type == "weekly":
        get_weekly_graph(time_test, y_test, y_pred, start_date_input)
    else:
        get_daily_graph(time_test, y_test, y_pred, start_date_input)

    return y_pred

def get_daily_graph(time_test, y_test, y_pred, start_date_input):

    if start_date_input == "":
        print("No date entered. Showing default daily visualization for 2025-01-01.")
        daily_real_load_vs_planned_load(time_test, y_test, y_pred)

    else:
        try:
            start_date = pd.to_datetime(start_date_input, format="%Y-%m-%d")

            if start_date < pd.to_datetime("2025-01-01") or start_date > pd.to_datetime("2025-12-31"):
                print("Date is outside the allowed range.")
                print("Showing default daily visualization for 2025-01-01.")
                daily_real_load_vs_planned_load(time_test, y_test, y_pred)

            else:
                print(f"Valid date entered.")
                print(f"Showing data for {start_date.date()}.")
                daily_real_load_vs_planned_load(time_test, y_test, y_pred, start_date)

        except ValueError:
            print("Invalid date format. Expected format is YYYY-MM-DD.")
            print("Showing default daily visualization for 2025-01-01.")
            daily_real_load_vs_planned_load(time_test, y_test, y_pred)
    return

def get_weekly_graph(time_test, y_test, y_pred, start_date_input):

    if start_date_input == "":
        print("No date entered. Showing default period from 2025-01-01 to 2025-01-07.")
        weekly_real_load_vs_planned_load(time_test, y_test, y_pred)

    else:
        try:
            start_date = pd.to_datetime(start_date_input, format="%Y-%m-%d")

            if start_date < pd.to_datetime("2025-01-01") or start_date > pd.to_datetime("2025-12-27"):
                print("Date is outside the allowed range of the 7-day forecast.")
                print("Showing default period from 2025-01-01 to 2025-01-07.")
                weekly_real_load_vs_planned_load(time_test, y_test, y_pred)

            else:
                end_date = start_date + pd.Timedelta(days=6)
                print(f"Valid date entered.")
                print(f"Showing data from {start_date.date()} to {end_date.date()}.")
                weekly_real_load_vs_planned_load(time_test, y_test, y_pred, start_date)

        except ValueError:
            print("Invalid date format. Expected format is YYYY-MM-DD.")
            print("Showing default period from 2025-01-01 to 2025-01-07.")
            weekly_real_load_vs_planned_load(time_test, y_test, y_pred)

    return