import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.pipelines import data_setup as ds
from src.pipelines import experiment_pipelines as ep


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "weekstatus": ["Weekday", "Weekend", "Weekday", "Weekend"],
            "day_of_week": ["Monday", "Sunday", "Tuesday", "Sunday"],
            "load_type": ["light", "medium", "light", "heavy"],
            "lagging_current_reactive.power_kvarh": [1.0, 2.0, 1.5, 2.5],
            "leading_current_reactive_power_kvarh": [0.1, 0.2, 0.3, 0.4],
            "co2(tco2)": [10.0, 11.0, 12.0, 13.0],
            "lagging_current_power_factor": [0.9, 0.85, 0.95, 0.8],
            "leading_current_power_factor": [0.1, 0.2, 0.15, 0.1],
            "nsm": [100, 200, 150, 250],
            "date": pd.to_datetime(
                ["2024-01-01 05:00:00", "2024-01-02 06:00:00", "2024-01-03 07:00:00", "2024-01-04 08:00:00"]
            ),
            "usage_kwh": [10, 20, 15, 18],
        }
    )


def test_feature_config_properties():
    cfg = ds.DEFAULT_FEATURE_CONFIG
    assert "date_hour" in cfg.date_feature_names
    assert "lagging_current_reactive.power_kvarh__div__leading_current_power_factor" in cfg.interaction_feature_names
    # numeric features for scaling joins base + date + interaction
    assert len(cfg.numeric_features_for_scaling) == len(cfg.numeric_base_features) + len(cfg.date_feature_names) + len(
        cfg.interaction_feature_names
    )


def test_infer_and_resolve_data_path(tmp_path):
    (tmp_path / "data" / "clean").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    csv_path = tmp_path / ds.DEFAULT_DATA_REL_PATH
    csv_path.write_text("date,usage_kwh\n2024-01-01,1\n")

    root = ds.infer_project_root(start=tmp_path / "src")
    assert root == tmp_path
    resolved = ds.resolve_data_path(project_root=root)
    assert resolved == csv_path


def test_load_clean_dataframe_uses_dataloader(monkeypatch, tmp_path):
    fake_df = pd.DataFrame({"date": ["2024-01-01"], "usage_kwh": [1], "weekstatus": ["Weekday"]})

    class DummyLoader:
        def __init__(self, path):
            self.input_path = path

        def load_data(self):
            return fake_df.copy()

    monkeypatch.setattr(ds, "DataLoader", DummyLoader)
    df = ds.load_clean_dataframe(data_path=tmp_path / "any.csv")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_build_feature_frame_success(sample_df):
    X, y = ds.build_feature_frame(sample_df, ds.DEFAULT_FEATURE_CONFIG)
    expected_cols = ds.DEFAULT_FEATURE_CONFIG.categorical_features + ds.DEFAULT_FEATURE_CONFIG.numeric_base_features + [
        "date"
    ]
    assert list(X.columns) == expected_cols
    assert y.tolist() == sample_df["usage_kwh"].tolist()


def test_build_feature_frame_missing_columns_raises(sample_df):
    df_missing = sample_df.drop(columns=["lagging_current_power_factor"])
    with pytest.raises(ValueError):
        ds.build_feature_frame(df_missing, ds.DEFAULT_FEATURE_CONFIG)


def test_build_feature_engineering_pipeline_adds_features(sample_df):
    pipeline = ep.build_feature_engineering_pipeline(ds.DEFAULT_FEATURE_CONFIG, drop_na=False, use_date_features=True)
    transformed = pipeline.fit_transform(sample_df)
    assert "date" not in transformed.columns
    assert "date_hour" in transformed.columns
    assert "lagging_current_reactive.power_kvarh__div__leading_current_power_factor" in transformed.columns


def test_build_preprocessor_output_shape(sample_df):
    fe_pipeline = ep.build_feature_engineering_pipeline(ds.DEFAULT_FEATURE_CONFIG)
    features = fe_pipeline.fit_transform(sample_df)
    preprocessor = ep.build_preprocessor(ds.DEFAULT_FEATURE_CONFIG)
    transformed = preprocessor.fit_transform(features)
    cat_cols = sum(features[c].nunique() for c in ds.DEFAULT_FEATURE_CONFIG.categorical_features)
    num_cols = len(ds.DEFAULT_FEATURE_CONFIG.numeric_base_features) + len(ds.DEFAULT_FEATURE_CONFIG.date_feature_names) + len(
        ds.DEFAULT_FEATURE_CONFIG.interaction_feature_names
    )
    expected_cols = cat_cols + num_cols
    assert transformed.shape == (len(sample_df), expected_cols)


def test_build_linear_pipeline_fits_and_predicts(sample_df):
    pipeline = ep.build_linear_pipeline(ds.DEFAULT_FEATURE_CONFIG)
    pipeline.fit(sample_df, sample_df["usage_kwh"])
    preds = pipeline.predict(sample_df)
    assert len(preds) == len(sample_df)


def test_build_xgb_pipeline_with_stubbed_regressor(sample_df, monkeypatch):
    class DummyXGB:
        def __init__(self, **kwargs):
            self.params = kwargs

        def fit(self, X, y):
            self.coef_ = np.zeros(X.shape[1])
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def get_params(self, deep=True):
            return self.params

        def set_params(self, **params):
            self.params.update(params)
            return self

    monkeypatch.setattr(ep.xgb, "XGBRegressor", DummyXGB)
    pipeline = ep.build_xgb_pipeline(ds.DEFAULT_FEATURE_CONFIG)
    pipeline.fit(sample_df, sample_df["usage_kwh"])
    preds = pipeline.predict(sample_df)
    assert len(preds) == len(sample_df)


def test_evaluate_regression_returns_metrics(sample_df, monkeypatch):
    class DummyPipe:
        def predict(self, X):
            return np.ones(len(X))

    monkeypatch.setattr(ep, "mean_squared_error", lambda y_true, y_pred, squared=False: float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))
    metrics = ep.evaluate_regression(DummyPipe(), sample_df, sample_df, sample_df["usage_kwh"], sample_df["usage_kwh"], "dummy")
    assert {"model", "rmse_train", "rmse_test", "mae_test", "r2_train", "r2_test"}.issubset(metrics.keys())


def test_cross_validate_pipeline_produces_summary(sample_df):
    simple_pipeline = Pipeline([("regressor", LinearRegression())])
    results, summary = ep.cross_validate_pipeline(
        simple_pipeline, sample_df[["lagging_current_reactive.power_kvarh", "nsm"]], sample_df["usage_kwh"], cv=2
    )
    assert "test_r2" in results
    assert list(summary.columns) == ["metric", "train_mean", "test_mean", "test_std"]


def test_run_xgb_grid_search_with_stubbed_regressor(sample_df, monkeypatch):
    class DummyXGB:
        def __init__(self, **kwargs):
            self.params = kwargs

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def get_params(self, deep=True):
            return self.params

        def set_params(self, **params):
            self.params.update(params)
            return self

    monkeypatch.setattr(ep.xgb, "XGBRegressor", DummyXGB)
    pipeline = ep.build_xgb_pipeline(ds.DEFAULT_FEATURE_CONFIG)
    grid = ep.run_xgb_grid_search(
        pipeline,
        sample_df,
        sample_df["usage_kwh"],
        param_grid={"regressor__n_estimators": [1], "regressor__max_depth": [2]},
        cv=2,
    )
    assert hasattr(grid, "best_params_")
    assert "regressor__n_estimators" in grid.best_params_


def test__load_features_reads_csv(tmp_path):
    csv_path = tmp_path / "features.csv"
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4],
            "f2": [10, 20, 30, 40],
            "usage_kwh": [0.5, 1.0, 1.5, 2.0],
        }
    )
    df.to_csv(csv_path, index=False)

    X_train, X_test, y_train, y_test = ep._load_features(path=str(csv_path), target="usage_kwh")
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.columns) == {"f1", "f2"}
    assert not y_train.isna().any()


def test__run_one_invalid_model_raises():
    with pytest.raises(ValueError):
        ep._run_one("unknown", None, None, None, None)


def test_main_respects_params_yaml(monkeypatch, tmp_path):
    calls = []

    def fake_load_features():
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        return X, X, y, y

    def fake_run_one(kind, *args):
        calls.append(kind)

    params_path = tmp_path / "params.yaml"
    params_path.write_text("train:\n  model_types:\n    - linear_regression\n    - xgboost\n")

    monkeypatch.setattr(ep, "_load_features", fake_load_features)
    monkeypatch.setattr(ep, "_run_one", fake_run_one)

    ep.main(params_path=str(params_path))

    assert calls == ["linear_regression", "xgboost"]
