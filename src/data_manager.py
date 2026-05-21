from typing import Any

import pandas as pd
import holidays as hd

def load_data() -> pd.DataFrame:
    file_path = "../data/data_table.xlsx"
    sheet_names = ["2025", "2024", "2023"]
    all_sheets = []
    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet, usecols="B:D", engine="openpyxl")

        df.columns = ["time", "planned_load", "real_load"]
        df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M", errors="coerce")
        df["planned_load"] = pd.to_numeric(df["planned_load"], errors="coerce")
        df["real_load"] = pd.to_numeric(df["real_load"], errors="coerce")
        all_sheets.append(df)

    data = pd.concat(all_sheets, ignore_index=True)
    data = data.dropna(subset=["time"])
    data = data.sort_values("time")
    data = data.reset_index(drop=True)

    return data

def city_temp_data() -> pd.DataFrame:
    year_of_temp = ["2025", "2024", "2023"]
    cities = ["zagreb", "split", "rijeka", "osijek", "zadar"]

    all_city_dfs = []

    for city in cities:
        yearly_dfs = []
        for year in year_of_temp:
            path_to_temp = "../temps/temps" + year + "/" + city + "Temps" + year + ".csv"
            df = pd.read_csv(path_to_temp)
            temp_col = city + "_temps"
            df = df[["date", "tavg"]].copy()
            df.columns = ["date", temp_col]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
            yearly_dfs.append(df)

        city_df = pd.concat(yearly_dfs)

        all_city_dfs.append(city_df)

    # merge city temperature dataframes by date
    data = all_city_dfs[0]
    for df in all_city_dfs[1:]:
        data = data.merge(df, on="date", how="outer")

    data = data.sort_values("date")
    data = data.reset_index(drop=True)

    return data

def merge_load_and_temps() -> pd.DataFrame:
    # making data frame for load
    load_df = load_data()

    # making data frame for temps
    city_temp_df = city_temp_data()

    load_df["date"] = pd.to_datetime(load_df["time"]).dt.normalize()
    load_df["hour"] = load_df["time"].dt.hour
    load_df["weekday"] = load_df["time"].dt.weekday
    load_df["month"] = load_df["time"].dt.month
    load_df["day_of_year"] = load_df["time"].dt.dayofyear
    load_df["year"] = load_df["time"].dt.year

    merged_df = load_df.merge(city_temp_df, on="date", how="left")

    cro_holidays = hd.HR(years=[2023, 2024, 2025])
    #adding workday indicator column
    merged_df["is_workday"] = ( (merged_df["date"].dt.weekday < 5) & (~merged_df["date"].dt.date.isin(cro_holidays)) )
    #adding weekend indicator column
    merged_df["is_weekend"] = (merged_df["date"].dt.weekday >= 5)
    merged_df = merged_df.drop(columns=["date", "planned_load"])

    return merged_df


def split_train_df_and_test_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    #getting dataframe with 2023, 2024 and 2025 data
    merged_df = merge_load_and_temps()

    #splitting into test and train dataframes
    train_df = merged_df[merged_df["year"].isin([2023, 2024])]
    test_df = merged_df[merged_df["year"] == 2025]

    return train_df, test_df