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