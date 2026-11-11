from src.load_data import load_raw
from src.clean import clean
from src.features import split_data
from src.train import train_models


def main():
    #1. Loading raw data
    print("1. Loading raw data")
    df = load_raw()

    #2. Cleaning
    print("2. Cleaning data")
    df = clean(df)

    #3. Splitting into train and test
    print("3. Splitting into train and test")
    X_train, X_test, y_train, y_test = split_data(df)

    #4. Training models
    print("4. Training models")
    models = train_models(X_train, y_train)

if __name__ == "__main__":
    main()