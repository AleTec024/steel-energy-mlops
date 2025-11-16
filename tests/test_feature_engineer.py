import os

import numpy as np
import pandas as pd
import pytest

from data.feature_engineer import (
    DateFeatureTransformer,
    FeatureEngineer,
    NumericInteractionTransformer,
    OutlierClipper,
)


def test_date_feature_transformer_produces_expected_columns():
    df = pd.DataFrame({"date": ["2024-01-01 06:00:00", "2024-01-02 12:30:00"]})
    transformer = DateFeatureTransformer(drop_original=True, add_cyclical=True, drop_na=True)

    transformed = transformer.fit_transform(df)

    expected_cols = [
        "date_hour",
        "date_dayofweek",
        "date_month",
        "date_dayofyear",
        "date_sin_hour",
        "date_cos_hour",
    ]
    assert expected_cols == transformer.get_feature_names_out().tolist()
    assert set(expected_cols) == set(transformed.columns)
    assert "date" not in transformed.columns
    assert np.isclose(transformed.iloc[0]["date_sin_hour"], 1.0)
    assert np.isclose(transformed.iloc[1]["date_cos_hour"], -1.0, atol=1e-6)


def test_date_feature_transformer_drop_na_rejects_invalid_dates():
    df = pd.DataFrame({"date": ["invalid-date", "2024-01-02 12:30:00"]})
    transformer = DateFeatureTransformer(drop_na=True)

    with pytest.raises(ValueError):
        transformer.fit_transform(df)


def test_numeric_interaction_transformer_generates_products_and_ratios():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 4.0], "c": [5.0, 6.0]})
    transformer = NumericInteractionTransformer(columns=["a", "b"], create_products=True, create_ratios=True)

    transformed = transformer.fit_transform(df)

    assert transformed["a__x__b"].tolist() == [0.0, 8.0]
    assert transformed["a__div__b"].tolist() == [0.0, 0.5]
    assert transformer.get_feature_names_out(["a", "b"]).tolist() == ["a", "b", "a__x__b", "a__div__b"]


def test_numeric_interaction_transformer_validates_columns():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    transformer = NumericInteractionTransformer(columns=["a", "missing"])

    with pytest.raises(KeyError):
        transformer.fit(df)


def test_outlier_clipper_limits_values_to_quantiles():
    df = pd.DataFrame({"x": [1.0, 2.0, 100.0, 3.0], "y": [5.0, 5.0, 5.0, 5.0]})
    clipper = OutlierClipper(lower_quantile=0.25, upper_quantile=0.75).fit(df)

    transformed = clipper.transform(df)

    lower, upper = df["x"].quantile([0.25, 0.75])
    assert transformed["x"].min() >= lower
    assert transformed["x"].max() <= upper
    assert transformed["y"].eq(5.0).all()
    assert set(clipper.get_feature_names_out()) == {"x", "y"}


def test_feature_engineer_run_drops_columns_splits_and_saves(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4],
            "f2": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
            "drop_me": [99, 98, 97, 96],
        }
    )
    fe = FeatureEngineer(features=["f1", "f2"], target="target", test_size=0.5, random_state=0, drop_columns=["drop_me"])

    saved = {}

    def fake_save_features(X, y, output_dir="data/interim"):
        output_dir = tmp_path / "features"
        os.makedirs(output_dir, exist_ok=True)
        path = output_dir / "features.csv"
        df_out = X.copy()
        df_out[fe.target] = y
        df_out.to_csv(path, index=False)
        saved["path"] = path

    monkeypatch.setattr(fe, "save_features", fake_save_features)

    X_train, X_test, y_train, y_test = fe.run(df, split=True)

    assert saved["path"].exists()
    assert "drop_me" not in X_train.columns
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_feature_engineer_cannot_drop_target_column():
    df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})
    fe = FeatureEngineer(features=["f1"], target="target", drop_columns=["target"])

    with pytest.raises(ValueError):
        fe.run(df)
