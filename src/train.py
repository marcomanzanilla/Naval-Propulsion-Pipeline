from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


def make_models(seed=42):
    """Create every model with its settings."""
    return {
        "baseline": DummyRegressor(strategy="mean"),
        "linear regression": LinearRegression(),
        "decision tree": DecisionTreeRegressor(random_state=seed),
        "random forest": RandomForestRegressor(n_estimators=100,
                                               random_state=seed, n_jobs=-1),
        "xgboost": MultiOutputRegressor(
            XGBRegressor(random_state=seed, n_jobs=-1)),
    }


def train_models(X_train, y_train, seed=42):
    """Train every model on the same training rows and return them by name."""
    models = make_models(seed)
    for model in models.values():
        model.fit(X_train, y_train)
    return models


if __name__ == "__main__":
    from sklearn.metrics import (root_mean_squared_error, mean_absolute_error)
    from load_data import load_raw
    from clean import clean
    from features import split_data

    df = clean(load_raw())
    X_train, X_test, y_train, y_test = split_data(df)

    models = train_models(X_train, y_train)

    print("Each pair below is (compressor, turbine)")
    for name, model in models.items():
        pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, pred, multioutput="raw_values")
        mae = mean_absolute_error(y_test, pred, multioutput="raw_values")
        print(f"{name:20s} RMSE: {rmse.round(5)}  MAE: {mae.round(5)}")