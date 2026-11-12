from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = ["compressor wear", "turbine wear"]


def score_models(models, X_test, y_test):
    """Test RMSE and MAE for every model, plus % improvement over the baseline"""

    rows = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, pred, multioutput="raw_values")
        mae = mean_absolute_error(y_test, pred, multioutput="raw_values")
        rows[name] = [rmse[0], rmse[1], mae[0], mae[1]]

    results = pd.DataFrame(rows, index=["compressor_rmse", "turbine_rmse",
                                        "compressor_mae", "turbine_mae"]).T
    base = results.loc["baseline"]
    results["compressor_improvement_pct"] = (1 - results["compressor_rmse"] / base["compressor_rmse"]) * 100
    results["turbine_improvement_pct"] = (1 - results["turbine_rmse"] / base["turbine_rmse"]) * 100
    return results.round({"compressor_rmse": 5, "turbine_rmse": 5,
                          "compressor_mae": 5, "turbine_mae": 5,
                          "compressor_improvement_pct": 5, "turbine_improvement_pct": 5})


def plot_model_comparison(results):
    """Create a bar chart of test RMSE for each model (lower is better)"""

    ax = results[["compressor_rmse", "turbine_rmse"]].plot.bar(figsize=(8, 5))
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Test error by model")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual(model, X_test, y_test):
    """Create a scatter plot of true vs. predicted wear for the forest; points on the red line are perfect"""

    pred = model.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i, ax in enumerate(axes):
        actual = y_test.iloc[:, i]
        ax.scatter(actual, pred[:, i], s=5, alpha=0.15)
        line = [actual.min(), actual.max()]
        ax.plot(line, line, color="red")
        ax.set_title(TARGET_NAMES[i])
        ax.set_xlabel("true decay state")
        ax.set_ylabel("predicted decay state")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_forest_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names):
    """Plots which sensors the forest relied on most (combined across both targets)"""

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    ax = importance.plot.barh(figsize=(8, 6))
    ax.set_xlabel("importance")
    ax.set_title("Random forest: which sensors mattered")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(models, X_test, y_test):
    """Run all evaluation steps: score, save the table, make the figures"""
    
    results = score_models(models, X_test, y_test)
    results.to_csv(REPORT_DIR / "model_results.csv")

    forest = models["random forest"]
    plot_model_comparison(results)
    plot_predicted_vs_actual(forest, X_test, y_test)
    plot_feature_importance(forest, X_test.columns)
    return results


if __name__ == "__main__":
    from load_data import load_raw
    from clean import clean
    from features import split_data
    from train import train_models

    df = clean(load_raw())
    X_train, X_test, y_train, y_test = split_data(df)
    models = train_models(X_train, y_train)
    print(evaluate(models, X_test, y_test))
