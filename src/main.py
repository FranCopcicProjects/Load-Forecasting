from data_loader import *

def main():
    load_file_path = "../data/data_table.xlsx"
    #making data frame for load
    ld = load_data(load_file_path)

    print(ld.head())

    #ld.to_csv("../data/cleaned_load_data.csv", index=False)

    #making data frame for temps
    td = temp_data()

    print(td.head())

if __name__ == "__main__":
    main()

