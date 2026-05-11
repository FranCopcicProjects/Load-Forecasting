from data_manager import *
from missing_data import *

def main():
    # making data frame for load
    #ld = load_data()

    #print(ld.head())

    # ld.to_csv("../data/cleaned_load_data.csv", index=False)

    # making data frame for temps
    #td = city_temp_data()

    #print(td.head())

    # weights of city temperatures
    zg_w, sp_w, ri_w, os_w, zd_w = 0.2, 0.2, 0.2, 0.2, 0.2

    md = merge_load_and_temps()
   #print(md.head(195))
    md.to_csv("../data/cleaned_load_data.csv", index=False)

    missing = find_missing_data()
    print(missing.head())
    #missing.to_excel("../data/missing_load_data.xlsx", index=False)

    missing_interval = find_missing_intervals()
    print(missing_interval.head())
    missing_interval.to_excel("../data/missing_load_data_intervals.xlsx", index=False)

if __name__ == "__main__":
    main()

