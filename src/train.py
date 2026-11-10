from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_models(X_train, y_train, seed=42):
    models = {
        "baseline": DummyRegressor(strategy="mean"),
        "linear regression": LinearRegression(),
        "decision tree": DecisionTreeRegressor(random_state=seed),
        "random forest": RandomForestRegressor(n_estimators=100,
                                               random_state=seed, n_jobs=-1),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


if __name__ == "__main__":
    from sklearn.metrics import root_mean_squared_error
    from load_data import load_raw
    from clean import clean
    from features import split_data

    df = clean(load_raw())
    X_train, X_test, y_train, y_test = split_data(df)

    models = train_models(X_train, y_train)

    for name, model in models.items():
        rmse = root_mean_squared_error(y_test, model.predict(X_test),
                                       multioutput="raw_values")
        print(f"{name:20s} RMSE (compressor, turbine): {rmse.round(5)}")