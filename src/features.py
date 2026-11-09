from sklearn.model_selection import train_test_split

TARGETS = ["gt_compressor_decay_state", "gt_turbine_decay_state"]


def split_data(df, test_size=0.2, seed=42):
    X = df.drop(columns=TARGETS)
    y = df[TARGETS]
    return train_test_split(X, y, test_size=test_size, random_state=seed)


if __name__ == "__main__":
    from load_data import load_raw
    from clean import clean

    df = clean(load_raw())
    X_train, X_test, y_train, y_test = split_data(df)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)