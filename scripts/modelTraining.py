import argparse
import os
import warnings
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pysr import PySRRegressor

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = [
    "workflow_type",
    "avg_file_size",
    "task_count",
    "edge_count",
    "workflow_depth",
]

TARGET_COMPUTE_TIME = "compute_time"
TARGET_TOTAL_POWER = "total_power"
MODELS_DIR = "models"
FIGURES_DIR = "figures"
CURRENT_MODELS_DIR = MODELS_DIR
CURRENT_FIGURES_DIR = FIGURES_DIR

ALL_MODEL_KEYS = ["lr", "pr", "er", "poly", "rf", "xgb", "gbr", "catboost", "lgbm", "sr"]

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    return df

def prepare_data(df: pd.DataFrame) -> tuple:
    X = df[FEATURE_COLUMNS].copy()
    y_ct = df[TARGET_COMPUTE_TIME].copy()
    y_tp = df[TARGET_TOTAL_POWER].copy()

    le = LabelEncoder()
    X["workflow_type"] = le.fit_transform(X["workflow_type"])

    print(f"\nWorkflow type classes: {list(le.classes_)}")
    print(f"\nFeatures shape: {X.shape}")
    print(f"Compute time target shape: {y_ct.shape}")
    print(f"Total power target shape: {y_tp.shape}")

    X_train_ct, X_test_ct, y_train_ct, y_test_ct = train_test_split(
        X, y_ct, test_size=0.2, random_state=42
    )
    X_train_tp, X_test_tp, y_train_tp, y_test_tp = train_test_split(
        X, y_tp, test_size=0.2, random_state=42
    )

    print(f"\nTrain size: {X_train_ct.shape[0]}, Test size: {X_test_ct.shape[0]}")

    return (
        X_train_ct, X_test_ct, y_train_ct, y_test_ct,
        X_train_tp, X_test_tp, y_train_tp, y_test_tp,
        le,
    )

def evaluate(y_true, y_pred, label: str) -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    mask = np.asarray(y_true) != 0
    mape = np.mean(np.abs((np.asarray(y_true)[mask] - np.asarray(y_pred)[mask])
                          / np.asarray(y_true)[mask])) * 100
    print(f"\n{label}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    print(f"  R²:   {r2:.4f}")
    return {"mse": mse, "mae": mae, "r2": r2, "rmse": rmse, "mape": mape}

def get_subdir(data_path: str, root_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    if base_name.startswith("merged_data_"):
        base_name = base_name[len("merged_data_"):]
    elif base_name == "merged_data":
        base_name = "default"
    return os.path.join(root_dir, base_name)

def get_models_subdir(data_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    if base_name.startswith("merged_data_"):
        base_name = base_name[len("merged_data_"):]
    elif base_name == "merged_data":
        base_name = "default"
    return os.path.join(MODELS_DIR, base_name)

def save_model(model, filename: str) -> None:
    os.makedirs(CURRENT_MODELS_DIR, exist_ok=True)
    path = os.path.join(CURRENT_MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {path}")

def load_model(filename: str, models_dir: str | None = None):
    base_dir = models_dir or CURRENT_MODELS_DIR
    path = os.path.join(base_dir, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

def save_feature_importance(lines: list[str], filename: str) -> None:
    os.makedirs(CURRENT_FIGURES_DIR, exist_ok=True)
    path = os.path.join(CURRENT_FIGURES_DIR, filename)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Feature importance saved → {path}")

def save_mae_rmse_csv(results_ct: dict, results_tp: dict) -> None:
    os.makedirs(CURRENT_FIGURES_DIR, exist_ok=True)
    model_name_map = {
        "lr": "LR", "pr": "PR", "er": "ER", "poly": "Poly",
        "rf": "RF", "xgb": "XGB", "gbr": "GBR",
        "catboost": "CatBoost", "lgbm": "LGBM", "sr": "SR",
    }
    keys = list(results_ct.keys())
    df = pd.DataFrame({
        "model":    [model_name_map[k] for k in keys],
        "mae_ct":   [results_ct[k]["mae"]  for k in keys],
        "rmse_ct":  [results_ct[k]["rmse"] for k in keys],
        "mape_ct":  [results_ct[k]["mape"] for k in keys],
        "mae_tp":   [results_tp[k]["mae"]  for k in keys],
        "rmse_tp":  [results_tp[k]["rmse"] for k in keys],
        "mape_tp":  [results_tp[k]["mape"] for k in keys],
    })
    path = os.path.join(CURRENT_FIGURES_DIR, "mae_rmse_comparison.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def save_actual_vs_predicted_csv(
    y_true,
    predictions: list,
    model_keys: list[str],
    filename: str,
) -> None:
    os.makedirs(CURRENT_FIGURES_DIR, exist_ok=True)
    df = pd.DataFrame({"actual": y_true.values})
    for key, y_pred in zip(model_keys, predictions):
        df[f"pred_{key}"] = y_pred
    path = os.path.join(CURRENT_FIGURES_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def _importance_lines_simple(feature_names: list[str], weights, header: str) -> list[str]:
    lines = [header, "-" * len(header)]
    for feat, w in zip(feature_names, weights):
        lines.append(f"  {feat}: {w:.6f}")
    return lines

def _importance_lines_poly(feature_names: list[str], poly, model) -> list[str]:
    poly_feat_names = poly.get_feature_names_out(feature_names)
    header = "Polynomial Regression — coefficients per expanded feature"
    lines = [header, "-" * len(header)]
    lines.append(f"  intercept: {model.intercept_:.6f}")
    for feat, coef in zip(poly_feat_names, model.coef_):
        lines.append(f"  {feat}: {coef:.6f}")
    return lines

def train_linear_regression(X_train, y_train, X_test, y_test, label: str, save_as: str) -> dict:
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, f"Linear Regression - {label}")
    print(f"  Coefficients: {dict(zip(FEATURE_COLUMNS, model.coef_))}")

    header = f"Linear Regression — coefficients ({label})"
    lines = _importance_lines_simple(FEATURE_COLUMNS, model.coef_, header)
    lines.insert(2, f"  intercept: {model.intercept_:.6f}")
    suffix = label.lower().replace(" ", "_")
    save_feature_importance(lines, f"feature_importance_lr_{suffix}.txt")

    save_model(model, save_as)
    return {"model": model, "y_pred": y_pred, **metrics}

def train_power_regression(X_train, y_train, X_test, y_test, label: str, save_as: str) -> dict:
    EPS = 1e-9

    X_train_agg = np.maximum(X_train.values.mean(axis=1), EPS)
    X_test_agg  = np.maximum(X_test.values.mean(axis=1), EPS)
    y_train_safe = np.maximum(y_train.values, EPS)

    log_X_train = np.log(X_train_agg).reshape(-1, 1)
    log_X_test  = np.log(X_test_agg).reshape(-1, 1)
    log_y_train = np.log(y_train_safe)

    model = LinearRegression()
    model.fit(log_X_train, log_y_train)

    log_y_pred = model.predict(log_X_test)
    y_pred = np.exp(log_y_pred)

    a = np.exp(model.intercept_)
    b = model.coef_[0]
    print(f"  Power Regression equation: y = {a:.6f} * x^{b:.6f}")

    metrics = evaluate(y_test, y_pred, f"Power Regression - {label}")

    suffix = label.lower().replace(" ", "_")
    header = f"Power Regression — equation ({label})"
    lines = [
        header,
        "-" * len(header),
        f"  a (scale):   {a:.6f}",
        f"  b (exponent): {b:.6f}",
        f"  equation: y = {a:.6f} * x_agg^{b:.6f}",
        "",
        "  Note: x_agg = row-wise mean of all feature columns",
    ]
    save_feature_importance(lines, f"feature_importance_pr_{suffix}.txt")

    save_model(model, save_as)
    return {"model": model, "y_pred": y_pred, "a": a, "b": b, **metrics}

def train_exponential_regression(X_train, y_train, X_test, y_test, label: str, save_as: str) -> dict:
    EPS = 1e-9

    X_train_agg = X_train.values.mean(axis=1).reshape(-1, 1)
    X_test_agg  = X_test.values.mean(axis=1).reshape(-1, 1)
    y_train_safe = np.maximum(y_train.values, EPS)

    log_y_train = np.log(y_train_safe)

    model = LinearRegression()
    model.fit(X_train_agg, log_y_train)

    log_y_pred = model.predict(X_test_agg)
    y_pred = np.exp(log_y_pred)

    a = np.exp(model.intercept_)
    b = model.coef_[0]
    print(f"  Exponential Regression equation: y = {a:.6f} * e^({b:.6f} * x)")

    metrics = evaluate(y_test, y_pred, f"Exponential Regression - {label}")

    suffix = label.lower().replace(" ", "_")
    header = f"Exponential Regression — equation ({label})"
    lines = [
        header,
        "-" * len(header),
        f"  a (scale):    {a:.6f}",
        f"  b (rate):     {b:.6f}",
        f"  equation: y = {a:.6f} * e^({b:.6f} * x_agg)",
        "",
        "  Note: x_agg = row-wise mean of all feature columns",
    ]
    save_feature_importance(lines, f"feature_importance_er_{suffix}.txt")

    save_model(model, save_as)
    return {"model": model, "y_pred": y_pred, "a": a, "b": b, **metrics}

def train_polynomial_regression(
    X_train, y_train, X_test, y_test, label: str,
    save_model_as: str, save_poly_as: str, degree: int = 2
) -> dict:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    metrics = evaluate(y_test, y_pred, f"Polynomial Regression (degree={degree}) - {label}")

    lines = _importance_lines_poly(FEATURE_COLUMNS, poly, model)
    lines[0] = f"Polynomial Regression (degree={degree}) — coefficients ({label})"
    suffix = label.lower().replace(" ", "_")
    save_feature_importance(lines, f"feature_importance_poly_{suffix}.txt")

    save_model(model, save_model_as)
    save_model(poly, save_poly_as)
    return {"model": model, "poly": poly, "y_pred": y_pred, **metrics}

def train_random_forest(X_train, y_train, X_test, y_test, label: str, save_as: str) -> dict:
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, f"Random Forest - {label}")
    print("  Feature importances:")
    for feat, imp in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"    {feat}: {imp:.4f}")

    header = f"Random Forest — feature importances ({label})"
    lines = _importance_lines_simple(FEATURE_COLUMNS, model.feature_importances_, header)
    suffix = label.lower().replace(" ", "_")
    save_feature_importance(lines, f"feature_importance_rf_{suffix}.txt")

    save_model(model, save_as)
    return {"model": model, "y_pred": y_pred, **metrics}

def train_xgboost(X_train, y_train, X_test, y_test, label: str, save_as: str) -> dict:
    model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, f"XGBoost - {label}")
    print("  Feature importances:")
    for feat, imp in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"    {feat}: {imp:.4f}")

    header = f"XGBoost — feature importances ({label})"
    lines = _importance_lines_simple(FEATURE_COLUMNS, model.feature_importances_, header)
    suffix = label.lower().replace(" ", "_")
    save_feature_importance(lines, f"feature_importance_xgb_{suffix}.txt")

    save_model(model, save_as)
    return {"model": model, "y_pred": y_pred, **metrics}

def train_gradient_boosting(X_train, y_train, X_test, y_test, label: str, save_as: str) -> dict:
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, f"Gradient Boosting - {label}")
    print("  Feature importances:")
    for feat, imp in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"    {feat}: {imp:.4f}")

    header = f"Gradient Boosting — feature importances ({label})"
    lines = _importance_lines_simple(FEATURE_COLUMNS, model.feature_importances_, header)
    suffix = label.lower().replace(" ", "_")
    save_feature_importance(lines, f"feature_importance_gbr_{suffix}.txt")

    save_model(model, save_as)
    return {"model": model, "y_pred": y_pred, **metrics}

def train_catboost(X_train, y_train, X_test, y_test, label: str, save_as: str) -> dict:
    model = CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, f"CatBoost - {label}")
    importances = model.get_feature_importance()
    print("  Feature importances:")
    for feat, imp in zip(FEATURE_COLUMNS, importances):
        print(f"    {feat}: {imp:.4f}")

    header = f"CatBoost — feature importances ({label})"
    lines = _importance_lines_simple(FEATURE_COLUMNS, importances, header)
    suffix = label.lower().replace(" ", "_")
    save_feature_importance(lines, f"feature_importance_catboost_{suffix}.txt")

    save_model(model, save_as)
    return {"model": model, "y_pred": y_pred, **metrics}

def train_lightgbm(X_train, y_train, X_test, y_test, label: str, save_as: str) -> dict:
    model = LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, f"LightGBM - {label}")
    importances = model.feature_importances_
    print("  Feature importances:")
    for feat, imp in zip(FEATURE_COLUMNS, importances):
        print(f"    {feat}: {imp:.4f}")

    header = f"LightGBM — feature importances ({label})"
    lines = _importance_lines_simple(FEATURE_COLUMNS, importances, header)
    suffix = label.lower().replace(" ", "_")
    save_feature_importance(lines, f"feature_importance_lgbm_{suffix}.txt")

    save_model(model, save_as)
    return {"model": model, "y_pred": y_pred, **metrics}

def train_symbolic_regression(X_train, y_train, X_test, y_test, label: str, save_as: str) -> dict:
    print(f"\nTraining Symbolic Regression for {label} (this may take a while)...")
    model = PySRRegressor(
        niterations=40,
        population_size=30,
        verbosity=0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, f"Symbolic Regression - {label}")
    equation_str = str(model.sympy())
    print(f"  Equation: {equation_str}")

    suffix = label.lower().replace(" ", "_")
    header = f"Symbolic Regression — discovered equation ({label})"
    lines = [
        header,
        "-" * len(header),
        f"  Best equation: {equation_str}",
        "",
        "  Feature variables used in equation:",
    ]
    for i, feat in enumerate(FEATURE_COLUMNS):
        lines.append(f"    x{i} = {feat}")
    save_feature_importance(lines, f"feature_importance_sr_{suffix}.txt")

    save_model(model, save_as)
    return {"model": model, "y_pred": y_pred, **metrics}

def print_summary(results_ct: dict, results_tp: dict) -> None:
    key_label_map = {
        "lr":       "Linear Regression",
        "pr":       "Power Regression",
        "er":       "Exponential Regression",
        "poly":     "Polynomial Regression",
        "rf":       "Random Forest",
        "xgb":      "XGBoost",
        "gbr":      "Gradient Boosting",
        "catboost": "CatBoost",
        "lgbm":     "LightGBM",
        "sr":       "Symbolic Regression",
    }
    keys = list(results_ct.keys())
    model_labels = [key_label_map[k] for k in keys]

    summary = pd.DataFrame({
        "Model":    model_labels,
        "CT R²":   [results_ct[k]["r2"]   for k in keys],
        "CT RMSE": [results_ct[k]["rmse"] for k in keys],
        "TP R²":   [results_tp[k]["r2"]   for k in keys],
        "TP RMSE": [results_tp[k]["rmse"] for k in keys],
    })

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)

def train_all(data_path: str, models: list[str] | None = None) -> None:
    global CURRENT_MODELS_DIR, CURRENT_FIGURES_DIR

    selected = models if models else ALL_MODEL_KEYS
    invalid = [m for m in selected if m not in ALL_MODEL_KEYS]
    if invalid:
        raise ValueError(f"Unknown model type(s): {invalid}. Choose from {ALL_MODEL_KEYS}.")
    print(f"\nTraining models: {selected}")

    CURRENT_MODELS_DIR = get_subdir(data_path, MODELS_DIR)
    CURRENT_FIGURES_DIR = get_subdir(data_path, FIGURES_DIR)
    os.makedirs(CURRENT_MODELS_DIR, exist_ok=True)
    os.makedirs(CURRENT_FIGURES_DIR, exist_ok=True)
    print(f"Models will be saved to: {CURRENT_MODELS_DIR}")
    print(f"Figures will be saved to: {CURRENT_FIGURES_DIR}")

    df = load_data(data_path)
    (
        X_train_ct, X_test_ct, y_train_ct, y_test_ct,
        X_train_tp, X_test_tp, y_train_tp, y_test_tp,
        label_encoder,
    ) = prepare_data(df)

    save_model(label_encoder, "workflow_type_label_encoder.pkl")

    def run_models(X_train, y_train, X_test, y_test, label_prefix, file_suffix):
        results = {}
        if "lr" in selected:
            results["lr"] = train_linear_regression(
                X_train, y_train, X_test, y_test,
                label_prefix, f"lr_{file_suffix}.pkl")
        if "pr" in selected:
            results["pr"] = train_power_regression(
                X_train, y_train, X_test, y_test,
                label_prefix, f"pr_{file_suffix}.pkl")
        if "er" in selected:
            results["er"] = train_exponential_regression(
                X_train, y_train, X_test, y_test,
                label_prefix, f"er_{file_suffix}.pkl")
        if "poly" in selected:
            results["poly"] = train_polynomial_regression(
                X_train, y_train, X_test, y_test,
                label_prefix,
                f"poly_{file_suffix}.pkl",
                f"poly_features_{file_suffix}.pkl")
        if "rf" in selected:
            results["rf"] = train_random_forest(
                X_train, y_train, X_test, y_test,
                label_prefix, f"rf_{file_suffix}.pkl")
        if "xgb" in selected:
            results["xgb"] = train_xgboost(
                X_train, y_train, X_test, y_test,
                label_prefix, f"xgb_{file_suffix}.pkl")
        if "gbr" in selected:
            results["gbr"] = train_gradient_boosting(
                X_train, y_train, X_test, y_test,
                label_prefix, f"gbr_{file_suffix}.pkl")
        if "catboost" in selected:
            results["catboost"] = train_catboost(
                X_train, y_train, X_test, y_test,
                label_prefix, f"catboost_{file_suffix}.pkl")
        if "lgbm" in selected:
            results["lgbm"] = train_lightgbm(
                X_train, y_train, X_test, y_test,
                label_prefix, f"lgbm_{file_suffix}.pkl")
        if "sr" in selected:
            results["sr"] = train_symbolic_regression(
                X_train, y_train, X_test, y_test,
                label_prefix, f"sr_{file_suffix}.pkl")
        return results

    print("\n" + "=" * 60)
    print("COMPUTE TIME MODELS")
    print("=" * 60)
    results_ct = run_models(X_train_ct, y_train_ct, X_test_ct, y_test_ct,
                            "Compute Time", "compute_time")

    print("\n" + "=" * 60)
    print("TOTAL POWER MODELS")
    print("=" * 60)
    results_tp = run_models(X_train_tp, y_train_tp, X_test_tp, y_test_tp,
                            "Total Power", "total_power")

    print_summary(results_ct, results_tp)

    print("\n" + "=" * 60)
    print("SAVING PLOT DATA TO CSV")
    print("=" * 60)

    active_keys = list(results_ct.keys())

    save_mae_rmse_csv(results_ct, results_tp)

    save_actual_vs_predicted_csv(
        y_test_ct,
        [results_ct[k]["y_pred"] for k in active_keys],
        active_keys,
        "actual_vs_predicted_compute_time.csv",
    )

    save_actual_vs_predicted_csv(
        y_test_tp,
        [results_tp[k]["y_pred"] for k in active_keys],
        active_keys,
        "actual_vs_predicted_total_power.csv",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression models on merged workflow data.")
    parser.add_argument("data_path", type=str, help="Path to the merged_data CSV file.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_KEYS,
        default=None,
        metavar="MODEL",
        help=(
            "One or more model types to train. "
            f"Choices: {ALL_MODEL_KEYS}. "
            "Defaults to all models if omitted. "
            "Example: --models rf xgb gbr catboost lgbm"
        ),
    )
    args = parser.parse_args()
    train_all(args.data_path, models=args.models)
