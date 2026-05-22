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
    x_train, y_train, time_train = split_features_and_target_value(train_df)
    #print(x_train.head())
    #print(y_train.head())
    x_test, y_test, time_test = split_features_and_target_value(test_df)
    #print(x_test.head())
    #print(y_test.head())

    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)

    print("Choose the type of forecast visualization.")
    print("Type 'daily' for a 1-day forecast visualization.")
    print("Type 'weekly' for a 7-day forecast visualization.")
    print("If the input is invalid or left empty, the default visualization will be used.")
    print("Default visualization period: 2025-01-01.")
    print()

    graph_type = input("Enter graph type: ")

    y_pred_lr = None

    if graph_type == "weekly":
        print("Enter a start date for the 7-day load forecast visualization.")
        print("The expected date format is: YYYY-MM-DD")
        print("Example of a valid input: 2025-03-15")
        print("The allowed date range is from 2025-01-01 to 2025-12-27.")
        print("Dates after 2025-12-27 are not allowed because a full 7-day forecast would not be available.")
        print("If the input is invalid or left empty, the default period will be used.")
        print("Default visualization period: 2025-01-01 to 2025-01-07.")
        print()
        start_date_input = input("Enter start date: ")

    else:
        print("Enter a start date for the daily load forecast visualization.")
        print("The expected date format is: YYYY-MM-DD")
        print("Example of a valid input: 2025-03-15")
        print("The allowed date range is from 2025-01-01 to 2025-12-31.")
        print()
        start_date_input = input("Enter start date: ")

    y_pred_lr = linear_regression(x_train, y_train, x_test, y_test)

    # get graphs for linear regression
    if graph_type == "weekly":
        get_weekly_graph(time_test, y_test, y_pred_lr, start_date_input)
    else:
        get_daily_graph(time_test, y_test, y_pred_lr, start_date_input)




if __name__ == "__main__":
    main()

