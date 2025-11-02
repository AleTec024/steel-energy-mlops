"""Helper functions to instantiate experimental pipelines for different models."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from inspect import signature

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.feature_engineer import (
    DateFeatureTransformer,
    NumericInteractionTransformer,
    OutlierClipper,
)
from src.pipelines.data_setup import FeatureConfig, DEFAULT_FEATURE_CONFIG


DEFAULT_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def build_feature_engineering_pipeline(
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    drop_na: bool = False,
    use_date_features: bool = True,
) -> Pipeline:
    steps = []

    if use_date_features:
        steps.append(
            (
                "date_features",
                DateFeatureTransformer(
                    datetime_col="date",
                    drop_original=True,
                    add_cyclical=True,
                    generate_features=True,
                    drop_na=drop_na,
                ),
            )
        )
        clip_columns = config.numeric_base_features + config.date_feature_names
    else:
        steps.append(
            (
                "drop_date",
                DateFeatureTransformer(
                    datetime_col="date",
                    drop_original=True,
                    add_cyclical=False,
                    generate_features=False,
                    drop_na=False,
                ),
            )
        )
        clip_columns = config.numeric_base_features

    steps.append(
        (
            "clip_outliers",
            OutlierClipper(
                columns=clip_columns,
                lower_quantile=0.01,
                upper_quantile=0.99,
            ),
        )
    )

    steps.append(
        (
            "interactions",
            NumericInteractionTransformer(
                columns=config.interaction_columns,
                create_ratios=True,
                create_products=False,
            ),
        )
    )

    return Pipeline(steps=steps)


def build_preprocessor(
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    use_date_features: bool = True,
) -> ColumnTransformer:
    numeric_features = list(config.numeric_base_features)
    if use_date_features:
        numeric_features += config.date_feature_names
    numeric_features += config.interaction_feature_names

    encoder_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in signature(OneHotEncoder).parameters:
        encoder_kwargs["sparse_output"] = False
    else:
        encoder_kwargs["sparse"] = False

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(**encoder_kwargs)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                config.categorical_features,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
        ],
        remainder="drop",
    )


def build_linear_pipeline(
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    drop_na: bool = False,
    use_date_features: bool = True,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                build_feature_engineering_pipeline(
                    config,
                    drop_na=drop_na,
                    use_date_features=use_date_features,
                ),
            ),
            (
                "preprocessor",
                build_preprocessor(config, use_date_features=use_date_features),
            ),
            ("regressor", LinearRegression()),
        ]
    )


def build_xgb_pipeline(
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    drop_na: bool = False,
    use_date_features: bool = True,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                build_feature_engineering_pipeline(
                    config,
                    drop_na=drop_na,
                    use_date_features=use_date_features,
                ),
            ),
            (
                "preprocessor",
                build_preprocessor(config, use_date_features=use_date_features),
            ),
            (
                "regressor",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def get_default_xgb_param_grid() -> Dict[str, list]:
    return {
        "regressor__n_estimators": [200, 400],
        "regressor__max_depth": [4, 6],
        "regressor__learning_rate": [0.03, 0.1],
        "regressor__subsample": [0.8, 1.0],
        "regressor__colsample_bytree": [0.7, 1.0],
    }


def evaluate_regression(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label: str,
) -> Dict[str, float]:
    return {
        "model": label,
        "rmse_train": mean_squared_error(y_train, pipeline.predict(X_train), squared=False),
        "rmse_test": mean_squared_error(y_test, pipeline.predict(X_test), squared=False),
        "mae_test": mean_absolute_error(y_test, pipeline.predict(X_test)),
        "r2_train": r2_score(y_train, pipeline.predict(X_train)),
        "r2_test": r2_score(y_test, pipeline.predict(X_test)),
    }


def cross_validate_pipeline(
    pipeline: Pipeline,
    feature_df: pd.DataFrame,
    target: pd.Series,
    *,
    cv: int = 5,
    scoring: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, list], pd.DataFrame]:
    scoring_dict = scoring or DEFAULT_SCORING
    results = cross_validate(
        pipeline,
        feature_df,
        target,
        cv=cv,
        scoring=scoring_dict,
        return_train_score=True,
    )

    summary = pd.DataFrame(
        {
            "metric": ["rmse", "mae", "r2"],
            "train_mean": [
                -results["train_rmse"].mean(),
                -results["train_mae"].mean(),
                results["train_r2"].mean(),
            ],
            "test_mean": [
                -results["test_rmse"].mean(),
                -results["test_mae"].mean(),
                results["test_r2"].mean(),
            ],
            "test_std": [
                results["test_rmse"].std(),
                results["test_mae"].std(),
                results["test_r2"].std(),
            ],
        }
    )
    return results, summary


def run_xgb_grid_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict[str, list]] = None,
    *,
    cv: int = 3,
    scoring: str = "neg_root_mean_squared_error",
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid or get_default_xgb_param_grid(),
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid

# =======================
# Orquestador para DVC
# =======================
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.env import load_env
load_env()

# Rutas de métricas que DVC espera
METRIC_PATHS = {
    "random_forest": "reports/metrics_rf.json",
    "linear_regression": "reports/metrics_linear.json",
    "xgboost": "reports/metrics_xgb.json",
}

def _ensure_reports():
    os.makedirs("reports/figures", exist_ok=True)

def _load_features(path="data/interim/features.csv", target="usage_kwh"):
    df = pd.read_csv(path)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    num_df = df.select_dtypes(include=["number"]).dropna(subset=[target]).copy()
    y = num_df[target]
    X = num_df.drop(columns=[target])
    if X.shape[1] == 0:
        raise ValueError("No quedaron columnas numéricas en X. Revisa el dataset/intermediate.")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def _run_one(kind, X_train, X_test, y_train, y_test):
    _ensure_reports()

    # Lazy imports para evitar ciclos
    if kind == "random_forest":
        from src.models.random_forest_model.model_trainer import ModelTrainer as RFTrainer
        trainer = RFTrainer()
    elif kind == "linear_regression":
        from src.models.linear_regression_model.model_trainer import ModelTrainer as LINTrainer
        trainer = LINTrainer()
    elif kind == "xgboost":
        from src.models.xgboost_model.model_trainer import ModelTrainer as XGBTrainer
        trainer = XGBTrainer()
    else:
        raise ValueError(f"Modelo no soportado: {kind}")

    print(f"[PIPELINE] Entrenando {kind}…")
    metrics = trainer.run(X_train, X_test, y_train, y_test, model_type=kind)

    metrics_path = METRIC_PATHS[kind]
    if not os.path.exists(metrics_path):
        import json
        with open(metrics_path, "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        print(f"[PIPELINE] Métricas {kind} guardadas en {metrics_path} (fallback).")


def main(params_path="params.yaml"):
    # Lee params.yaml si existe; por defecto corre los 3
    try:
        import yaml
        with open(params_path, "r") as f:
            P = yaml.safe_load(f) or {}
        model_types = P.get("train", {}).get("model_types") or \
                      ["random_forest", "linear_regression", "xgboost"]
    except Exception:
        model_types = ["random_forest", "linear_regression", "xgboost"]

    X_train, X_test, y_train, y_test = _load_features()
    for kind in model_types:
        _run_one(kind, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
