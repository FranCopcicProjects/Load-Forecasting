import pandas as pd
from data_manager import *

def find_missing_data()->pd.DataFrame:
    curr_df = load_data()
    full_time_range = pd.date_range(
        start = curr_df["time"].min(),
        end = curr_df["time"].max(),
        freq = "h"
    )

    all_dates_df = pd.DataFrame({"datetime" : full_time_range})

    merged_df = all_dates_df.merge(curr_df, left_on="datetime", right_on="time", how="left")

    missing_df = merged_df[ merged_df["planned_load"].isna() & merged_df["real_load"].isna() ].copy()

    missing_df["date"] = missing_df["datetime"].dt.date
    missing_df["hour"] = missing_df["datetime"].dt.time

    missing_df = missing_df[ ["datetime", "date", "hour", "planned_load", "real_load"] ]

    return missing_df

import pandas as pd


def find_missing_intervals() -> pd.DataFrame:
    df = find_missing_data().copy()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    intervals = []

    start_time = df.loc[0, "datetime"]
    previous_time = df.loc[0, "datetime"]

    for i in range(1, len(df)):
        current_time = df.loc[i, "datetime"]

        if current_time - previous_time != pd.Timedelta(hours=1):

            end_time = previous_time

            interval_length = int( (end_time - start_time) / pd.Timedelta(hours=1) ) + 1

            intervals.append({"beginning_of_missing_data": start_time, "end_of_missing_data": end_time, "interval_length": interval_length})

            start_time = current_time

        previous_time = current_time

    end_time = previous_time

    interval_length = int( (end_time - start_time) / pd.Timedelta(hours=1) ) + 1

    intervals.append({ "beginning_of_missing_data": start_time, "end_of_missing_data": end_time, "interval_length": interval_length})

    result_df = pd.DataFrame(intervals)

    return result_df