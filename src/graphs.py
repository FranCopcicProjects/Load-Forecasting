import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def get_daily_graph(time_test, y_test, y_pred, start_date_input):
    #input validation
    if start_date_input == "":
        print("No date entered. Showing default daily visualization for 2025-01-02.")
        daily_real_load_vs_planned_load(time_test, y_test, y_pred)

    else:
        try:
            start_date = pd.to_datetime(start_date_input, format="%Y-%m-%d")

            if start_date < pd.to_datetime("2025-01-02") or start_date > pd.to_datetime("2025-12-31"):
                print("Date is outside the allowed range.")
                print("Showing default daily visualization for 2025-01-02.")
                daily_real_load_vs_planned_load(time_test, y_test, y_pred)

            else:
                print(f"Valid date entered.")
                print(f"Showing data for {start_date.date()}.")
                daily_real_load_vs_planned_load(time_test, y_test, y_pred, start_date)

        except ValueError:
            print("Invalid date format. Expected format is YYYY-MM-DD.")
            print("Showing default daily visualization for 2025-01-02.")
            daily_real_load_vs_planned_load(time_test, y_test, y_pred)
    return

def get_weekly_graph(time_test, y_test, y_pred, start_date_input):
    #input validation
    if start_date_input == "":
        print("No date entered. Showing default period from 2025-01-02 to 2025-01-07.")
        weekly_real_load_vs_planned_load(time_test, y_test, y_pred)

    else:
        try:
            start_date = pd.to_datetime(start_date_input, format="%Y-%m-%d")

            if start_date < pd.to_datetime("2025-01-02") or start_date > pd.to_datetime("2025-12-27"):
                print("Date is outside the allowed range of the 7-day forecast.")
                print("Showing default period from 2025-01-02 to 2025-01-07.")
                weekly_real_load_vs_planned_load(time_test, y_test, y_pred)

            else:
                end_date = start_date + pd.Timedelta(days=6)
                print(f"Valid date entered.")
                print(f"Showing data from {start_date.date()} to {end_date.date()}.")
                weekly_real_load_vs_planned_load(time_test, y_test, y_pred, start_date)

        except ValueError:
            print("Invalid date format. Expected format is YYYY-MM-DD.")
            print("Showing default period from 2025-01-02 to 2025-01-07.")
            weekly_real_load_vs_planned_load(time_test, y_test, y_pred)

    return

def weekly_real_load_vs_planned_load(time, y_true, y_pred, start_date="2025-01-02"):
    df = pd.DataFrame({"time": pd.to_datetime(time), "real_load": y_true, "predicted_load": y_pred})

    start_date = pd.to_datetime(start_date)
    end_date = start_date + pd.Timedelta(days=6)

    df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]

    plt.figure(figsize=(15,6))

    #real load
    plt.scatter(df["time"], df["real_load"], label="Real Load", color="blue", s=10)

    plt.scatter(df["time"], df["predicted_load"], label="Predicted Load", color="red", s=10)

    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def daily_real_load_vs_planned_load(time, y_true, y_pred, start_date="2025-01-02"):

    df = pd.DataFrame({"time": pd.to_datetime(time), "real_load": y_true, "predicted_load": y_pred})

    start_date = pd.to_datetime(start_date)

    end_date = start_date + pd.Timedelta(hours=23)

    df = df[ (df["time"] >= start_date) & (df["time"] <= end_date) ]

    plt.figure(figsize=(15,6))

    #real load
    plt.scatter(df["time"], df["real_load"], label="Real Load", color="blue", s=10)

    #predicted load
    plt.scatter(df["time"], df["predicted_load"],label="Predicted Load", color="red", s=10)

    plt.title(f"Daily Load Forecast for {start_date.date()}")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter( mdates.DateFormatter('%H:%M') )
    plt.gca().xaxis.set_major_locator( mdates.HourLocator(interval=1) )
    plt.show()

    return

