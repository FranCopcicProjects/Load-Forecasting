import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    sheet_names = ["2025", "2024", "2023"]
    all_sheets = []
    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet, usecols="B:D", engine="openpyxl")

        df.columns = ["time", "planned_load", "real_load"]
        df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M", errors="coerce")
        df["planned_load"] = pd.to_numeric(df["planned_load"], errors="coerce")
        df["real_load"] = pd.to_numeric(df["real_load"], errors="coerce")
        df["year"] = int(sheet)
        all_sheets.append(df)

    data = pd.concat(all_sheets, ignore_index=True)
    data = data.dropna(subset=["time"])
    data = data.sort_values("time")
    data = data.reset_index(drop=True)

    return data

def temp_data() -> pd.DataFrame:
    year_of_temp = ["2025", "2024", "2023"]
    citys = ["zagreb", "split", "rijeka", "osijek", "zadar"]

    all_temps = []

    for year in year_of_temp:
        for city in citys:
            path_to_temp = "../temps/temps" + year + "/" + city + "Temps" + year + ".csv"
            df = pd.read_csv(path_to_temp)
            df = df[["date", "tavg"]].copy()
            df.columns = ["date", "temp_avg"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["temp_avg"] = pd.to_numeric(df["temp_avg"], errors="coerce")
            df["city"] = city
            all_temps.append(df)


    data = pd.concat(all_temps, ignore_index=True)
    data = data.dropna(subset=["date"])
    data = data.sort_values("date")
    data = data.reset_index(drop=True)

    return data