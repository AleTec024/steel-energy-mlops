"""Helper functions to instantiate experimental pipelines for different models."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import argparse
import json
import yaml
import mlflow
import mlflow.sklearn
from pathlib import Path
from src.utils.env import load_env

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    
try:
    from src.data.feature_engineer import (
        DateFeatureTransformer,
        NumericInteractionTransformer,
        OutlierClipper,
    )
    _HAVE_FE = True
except Exception:
    # Fallback a transformadores identidad si no existen
    from sklearn.base import BaseEstimator, TransformerMixin
    class _Identity(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None): return self
        def transform(self, X): return X
    DateFeatureTransformer = NumericInteractionTransformer = OutlierClipper = _Identity
    _HAVE_FE = False

from src.pipelines.data_setup import FeatureConfig, DEFAULT_FEATURE_CONFIG



DEFAULT_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


#def build_feature_engineering_pipeline(
#    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
#) -> Pipeline:
#    return Pipeline(
#        steps=[
#            (
#                "date_features",
#                DateFeatureTransformer(datetime_col="date", drop_original=True, add_cyclical=True),
#           ),
#          (
#                "clip_outliers",
#                OutlierClipper(
#                    columns=config.numeric_base_features + config.date_feature_names,
#                    lower_quantile=0.01,
#                    upper_quantile=0.99,
#                ),
#            ),
#            (
#                "interactions",
#                NumericInteractionTransformer(
#                    columns=config.interaction_columns,
#                    create_ratios=True,
#                    create_products=False,
#                ),
#            ),
#        ]
#    )

def build_feature_engineering_pipeline(
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> Pipeline:
    # Si no tenemos los transformadores reales, usamos identidad (no-op)
    if not _HAVE_FE:
        from sklearn.pipeline import Pipeline
        from sklearn.base import BaseEstimator, TransformerMixin
        class _Identity(TransformerMixin, BaseEstimator):
            def fit(self, X, y=None): return self
            def transform(self, X): return X
        return Pipeline(steps=[("noop", _Identity())])

    return Pipeline(
        steps=[
            ("date_features", DateFeatureTransformer(datetime_col="date", drop_original=True, add_cyclical=True)),
            ("clip_outliers", OutlierClipper(
                columns=config.numeric_base_features + config.date_feature_names,
                lower_quantile=0.01, upper_quantile=0.99,
            )),
            ("interactions", NumericInteractionTransformer(
                columns=config.interaction_columns, create_ratios=True, create_products=False,
            )),
        ]
    )


def build_preprocessor(config: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                config.categorical_features,
            ),
            (
                "numeric",
                Pipeline([("scaler", StandardScaler())]),
                config.numeric_features_for_scaling,
            ),
        ],
        remainder="drop",
    )


def build_linear_pipeline(config: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", build_feature_engineering_pipeline(config)),
            ("preprocessor", build_preprocessor(config)),
            ("regressor", LinearRegression()),
        ]
    )


def build_xgb_pipeline(config: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", build_feature_engineering_pipeline(config)),
            ("preprocessor", build_preprocessor(config)),
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
        "rmse_train": mean_squared_error(y_train, pipeline.predict(X_train)),
        "rmse_test": mean_squared_error(y_test, pipeline.predict(X_test)),
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

def _load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_data_by_schema(params: dict):
    # Prioridad: processed_path si existe; si no, raw_path
    data_path = params.get("data", {}).get("processed_path") or params.get("data", {}).get("raw_path")
    target_col = params.get("features", {}).get("target", "usage_kwh")
    selected = params.get("features", {}).get("selected", [])

    if not data_path:
        raise ValueError("No se definió data.processed_path ni data.raw_path en params.yaml")

    df = pd.read_csv(data_path)
    missing = [c for c in selected + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en {data_path}: {missing}")

    X = df[selected].copy()
    y = df[target_col].copy()
    return df, X, y, data_path, target_col, selected

def _build_pipeline_by_schema(model_type: str, train_cfg: dict) -> Pipeline:
    # Aquí no usamos tu preprocesamiento complejo porque tus columnas selected ya están “listas”.
    if model_type == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost no está instalado (pip install xgboost).")
        cfg = train_cfg.get("xgboost", {})
        model = XGBRegressor(
            n_estimators=int(cfg.get("n_estimators", 200)),
            learning_rate=float(cfg.get("learning_rate", 0.1)),
            max_depth=int(cfg.get("max_depth", 6)),
            subsample=float(cfg.get("subsample", 0.8)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
            eval_metric=cfg.get("eval_metric", "rmse"),
            random_state=int(cfg.get("random_state", 42)),
            n_jobs=int(cfg.get("n_jobs", -1)),
        )
    elif model_type == "random_forest":
        cfg = train_cfg.get("random_forest", {})
        model = RandomForestRegressor(
            n_estimators=int(cfg.get("n_estimators", 300)),
            max_depth=cfg.get("max_depth", None),
            min_samples_split=int(cfg.get("min_samples_split", 2)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 1)),
            max_features=cfg.get("max_features", None),
            random_state=int(cfg.get("random_state", 42)),
            n_jobs=int(cfg.get("n_jobs", -1)),
        )
    elif model_type == "linear_regression":
        cfg = train_cfg.get("linear_regression", {})
        # n_jobs no aplica a LinearRegression clásico; lo ignoramos si viene
        model = LinearRegression(fit_intercept=bool(cfg.get("fit_intercept", True)))
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    return Pipeline([("regressor", model)])

def run_with_mlflow(params_path: str = "params.yaml"):
    env = load_env()
    params = _load_params(params_path)

    # Lee esquema
    model_type = params.get("train", {}).get("model_type", "xgboost")
    cv_folds   = int(params.get("train", {}).get("cv_folds", 5))

    # MLflow
    if env.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(env["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(env.get("EXPERIMENT_NAME", "steel-energy"))

    # Datos
    df, X, y, data_path, target_col, selected = _load_data_by_schema(params)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,  # puedes moverlo a params.train si gustas
        random_state=42
    )

    # Pipeline por schema
    pipe = _build_pipeline_by_schema(model_type, params.get("train", {}))

    # CV opcional (si > 0)
    cv_summary = None
    if cv_folds and cv_folds > 1:
        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
        cv_res = cross_validate(pipe, X, y, cv=cv_folds, scoring=scoring, return_train_score=True)
        # resumen
        import numpy as np, pandas as pd
        cv_summary = pd.DataFrame({
            "metric": ["rmse", "mae", "r2"],
            "train_mean": [-cv_res["train_rmse"].mean(), -cv_res["train_mae"].mean(),  cv_res["train_r2"].mean()],
            "test_mean":  [-cv_res["test_rmse"].mean(),  -cv_res["test_mae"].mean(),   cv_res["test_r2"].mean()],
            "test_std":   [ cv_res["test_rmse"].std(),    cv_res["test_mae"].std(),     cv_res["test_r2"].std()],
        })

    # Fit + evaluación final
    pipe.fit(X_train, y_train)
    metrics = evaluate_regression(pipe, X_train, X_test, y_train, y_test, label=model_type)

    # Log en MLflow
    with mlflow.start_run(run_name=f"train_{model_type}") as run:
        run_id = run.info.run_id

        # Tags y params de trazabilidad
        mlflow.set_tags({
            "stage": env.get("ENV", "local"),
            "data_path": data_path,
            "target": target_col,
            "n_features": len(selected),
        })
        # HParams del bloque de modelo
        for k, v in (params.get("train", {}).get(model_type, {}) or {}).items():
            mlflow.log_param(f"{model_type}_{k}", v)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("cv_folds", cv_folds)

        # Métricas
        mlflow.log_metrics({
            "rmse_train": metrics["rmse_train"],
            "rmse_test":  metrics["rmse_test"],
            "mae_test":   metrics["mae_test"],
            "r2_train":   metrics["r2_train"],
            "r2_test":    metrics["r2_test"],
        })

        # Artefactos
        reports = Path("reports"); reports.mkdir(parents=True, exist_ok=True)
        # guarda métricas y columnas usadas
        (reports / "used_columns.txt").write_text("\n".join(selected), encoding="utf-8")
        mlflow.log_artifact(str(reports / "used_columns.txt"))

        import json
        metrics_path = reports / f"metrics_{model_type}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

        if cv_summary is not None:
            cv_path = reports / f"cv_summary_{model_type}.csv"
            cv_summary.to_csv(cv_path, index=False)
            mlflow.log_artifact(str(cv_path))

        # Modelo (pipeline completo)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(f"✅ Run {run_id} | {model_type} | rmse_test={metrics['rmse_test']:.4f} | r2_test={metrics['r2_test']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()
    run_with_mlflow(args.params)
