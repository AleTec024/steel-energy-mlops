"""Helper functions to instantiate experimental pipelines for different models."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
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
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "date_features",
                DateFeatureTransformer(datetime_col="date", drop_original=True, add_cyclical=True),
            ),
            (
                "clip_outliers",
                OutlierClipper(
                    columns=config.numeric_base_features + config.date_feature_names,
                    lower_quantile=0.01,
                    upper_quantile=0.99,
                ),
            ),
            (
                "interactions",
                NumericInteractionTransformer(
                    columns=config.interaction_columns,
                    create_ratios=True,
                    create_products=False,
                ),
            ),
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

