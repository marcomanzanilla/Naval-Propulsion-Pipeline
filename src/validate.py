import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import root_mean_squared_error

from load_data import load_raw
from clean import clean
from features import TARGETS
from train import make_models

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def run_cv(df, splitter, groups=None, seed=42):
    """Train and score every model on every fold. One row is used per fold and model."""
    X = df.drop(columns=TARGETS)
    y = df[TARGETS]
    rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        # Creating the fold's training rows and test rows
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        held_out = groups.iloc[test_idx].iloc[0] if groups is not None else None

        # Train new models for every fold and time them
        for name, model in make_models(seed).items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start

            start = time.perf_counter()
            pred = model.predict(X_test)
            predict_seconds = time.perf_counter() - start

            rmse = root_mean_squared_error(y_test, pred, multioutput="raw_values")
            rows.append({"fold": fold, "held_out_lever": held_out, "model": name,
                         "compressor_rmse": rmse[0], "turbine_rmse": rmse[1],
                         "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
    return pd.DataFrame(rows)


def summarize(results):
    """Mean and standard deviation across folds for each model."""
    cols = ["compressor_rmse", "turbine_rmse", "fit_seconds", "predict_seconds"]
    summary = results.groupby("model", sort=False)[cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.round(5)


if __name__ == "__main__":
    df = clean(load_raw())

    #5-fold cross-validation with random folds
    random_cv = run_cv(df, KFold(n_splits=5, shuffle=True, random_state=42))
    random_cv.to_csv(REPORT_DIR / "cv_random_folds.csv", index=False)
    print("5-fold cross-validation (random folds):")
    print(summarize(random_cv))

    # Hold out one whole throttle setting per fold
    throttle_cv = run_cv(df, LeaveOneGroupOut(), groups=df["lever_position"])
    throttle_cv.to_csv(REPORT_DIR / "cv_held_out_throttle.csv", index=False)
    print("\nLeave-one-throttle-out cross-validation:")
    print(summarize(throttle_cv))

    print("\nRandom forest on each unseen throttle setting:")
    forest = throttle_cv[throttle_cv["model"] == "random forest"]
    print(forest[["held_out_lever", "compressor_rmse", "turbine_rmse"]]
          .round(5).to_string(index=False))