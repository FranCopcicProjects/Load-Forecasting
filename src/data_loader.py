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