from src.load_data import load_raw
from src.clean import clean
from src.features import split_data
from src.train import train_models
from src.evaluate import evaluate


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

    #5. Evaluating models
    print("5. Evaluating models")
    results = evaluate(models, X_test, y_test)
    print("Evaluation results:\n", results)
    print("Pipeline completed successfully. Generated figures are in reports/figures")

if __name__ == "__main__":
    main()