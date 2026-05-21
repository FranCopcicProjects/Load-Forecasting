from data_manager import *
from missing_data import *
from models import *

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

    #md = merge_load_and_temps()
    #print(md.head())
    #md.to_csv("../data/cleaned_load_data.csv", index=False)

    #missing = find_missing_data()
    #print(missing.head())
    #missing.to_excel("../data/missing_load_data.xlsx", index=False)

    #missing_interval = find_missing_intervals()
    #print(missing_interval.head())
    #missing_interval.to_excel("../data/missing_load_data_intervals.xlsx", index=False)

    train_df, test_df = split_train_df_and_test_df()
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    #print(train_df.head())
    #print(test_df.head())
    x_train, y_train = split_features_and_target_value(train_df)
    #print(x_train.head())
    #print(y_train.head())
    x_test, y_test = split_features_and_target_value(test_df)
    #print(x_test.head())
    #print(y_test.head())

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)



if __name__ == "__main__":
    main()

