import pandas as pd


def load_data():
    file_path = '/home/lxf/data/MedicalQA/train.csv'
    return pd.read_csv(file_path)


if __name__ == '__main__':
    init_data = load_data()
    print(init_data.head())