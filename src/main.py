from data_manager import *

def main():
    #making data frame for load
    ld = load_data()

    #print(ld.head())

    #ld.to_csv("../data/cleaned_load_data.csv", index=False)

    #making data frame for temps
    td = temp_data()

    print(td.head())

    #weights of city temperatures
    zg_w, sp_w, ri_w, os_w, zd_w = 0.2, 0.2, 0.2, 0.2, 0.2


if __name__ == "__main__":
    main()

