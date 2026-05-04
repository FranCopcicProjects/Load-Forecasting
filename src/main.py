from data_loader import load_data

def main():
    file_path = "../data/data_table.xlsx"

    data = load_data(file_path)

    data.to_csv("../data/cleaned_load_data.csv", index=False)

if __name__ == "__main__":
    main()

