import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    """Expande una columna temporal en atributos derivados, con control de NaN."""

    def __init__(
        self,
        datetime_col: str = "date",
        drop_original: bool = True,
        add_cyclical: bool = True,
        drop_na: bool = False,
        clip_outliers: bool = False,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ):
        self.datetime_col = datetime_col
        self.drop_original = drop_original
        self.add_cyclical = add_cyclical
        self.drop_na = drop_na
        self.clip_outliers = clip_outliers
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.feature_names_: Optional[Sequence[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_column(X)
        return self

    def transform(self, X: pd.DataFrame):
        self._validate_column(X)
        df = X.copy()
        dt = pd.to_datetime(df[self.datetime_col], errors="coerce")

        features = pd.DataFrame(
            {
                f"{self.datetime_col}_hour": dt.dt.hour.astype(np.float32),
                f"{self.datetime_col}_dayofweek": dt.dt.dayofweek.astype(np.float32),
                f"{self.datetime_col}_month": dt.dt.month.astype(np.float32),
                f"{self.datetime_col}_dayofyear": dt.dt.dayofyear.astype(np.float32),
            },
            index=df.index,
        )

        if self.add_cyclical:
            hour = features[f"{self.datetime_col}_hour"].astype(float)
            features[f"{self.datetime_col}_sin_hour"] = np.sin(2 * np.pi * hour / 24)
            features[f"{self.datetime_col}_cos_hour"] = np.cos(2 * np.pi * hour / 24)

        if self.drop_na:
            processed_features = features.astype(np.float32)
        else:
            fill_values = {
                f"{self.datetime_col}_hour": -1.0,
                f"{self.datetime_col}_dayofweek": -1.0,
                f"{self.datetime_col}_month": -1.0,
                f"{self.datetime_col}_dayofyear": -1.0,
            }
            if self.add_cyclical:
                fill_values[f"{self.datetime_col}_sin_hour"] = 0.0
                fill_values[f"{self.datetime_col}_cos_hour"] = 0.0
            processed_features = features.fillna(fill_values).astype(np.float32)

        if self.clip_outliers:
            if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
                raise ValueError("Los cuantiles deben cumplir 0 <= lower < upper <= 1")
            lower_bounds = processed_features.quantile(self.lower_quantile)
            upper_bounds = processed_features.quantile(self.upper_quantile)
            processed_features = processed_features.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

        if self.drop_original:
            df = df.drop(columns=[self.datetime_col])

        self.feature_names_ = list(processed_features.columns)
        return pd.concat([df, processed_features], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_ or [])

    def _validate_column(self, X: pd.DataFrame):
        if self.datetime_col not in X.columns:
            raise KeyError(f"[ERROR] La columna '{self.datetime_col}' no existe en el DataFrame.")


class NumericInteractionTransformer(BaseEstimator, TransformerMixin):
    """Genera interacciones simples entre columnas numericas."""

    def __init__(
        self,
        columns: Optional[Iterable[str]] = None,
        create_ratios: bool = True,
        create_products: bool = False,
    ):
        self.columns = tuple(columns) if columns is not None else None
        self.create_ratios = create_ratios
        self.create_products = create_products
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_columns_ = self._infer_numeric_columns(X)
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        new_features: list[str] = []

        for i, col_i in enumerate(self.numeric_columns_):
            for j, col_j in enumerate(self.numeric_columns_):
                if j <= i:
                    continue

                if self.create_products:
                    prod_name = f"{col_i}__x__{col_j}"
                    df[prod_name] = df[col_i] * df[col_j]
                    new_features.append(prod_name)

                if self.create_ratios:
                    ratio_name = f"{col_i}__div__{col_j}"
                    df[ratio_name] = df[col_i] / df[col_j].replace({0: np.nan})
                    df[ratio_name] = df[ratio_name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    new_features.append(ratio_name)

        self.feature_names_ = new_features
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features or []) + self.feature_names_)

    def _infer_numeric_columns(self, X: pd.DataFrame) -> Sequence[str]:
        if self.columns is not None:
            missing = sorted(set(self.columns) - set(X.columns))
            if missing:
                raise KeyError(f"[ERROR] Columnas faltantes en NumericInteractionTransformer: {missing}")
            return list(self.columns)

        return list(X.select_dtypes(include=[np.number]).columns)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Recorta valores extremos usando cuantiles por columna."""

    def __init__(self, columns: Optional[Iterable[str]] = None, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("Los cuantiles deben cumplir 0 <= lower < upper <= 1")

        self.columns = tuple(columns) if columns is not None else None
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or list(X.select_dtypes(include=[np.number]).columns)
        self.clip_values_ = {}
        for column in cols:
            q_low = X[column].quantile(self.lower_quantile)
            q_high = X[column].quantile(self.upper_quantile)
            self.clip_values_[column] = (q_low, q_high)
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        for column, (low, high) in self.clip_values_.items():
            if column not in df.columns:
                continue
            df[column] = df[column].clip(lower=low, upper=high)
        return df

    def get_feature_names_out(self, input_features=None):
        keys = list(self.clip_values_.keys())
        return np.asarray(list(input_features or keys))


class FeatureEngineer:
    """
    Handles feature selection, creation, and dataset splitting
    for model training.
    """

    def __init__(self, features: list, target: str, test_size: float = 0.2, random_state: int = 42):
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def select_features(self, df: pd.DataFrame):
        """
        Select feature and target columns from the preprocessed dataset.
        """
        print("[INFO] Selecting features and target...")

        missing_cols = [f for f in self.features + [self.target] if f not in df.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] Missing columns in dataset: {missing_cols}")

        X = df[self.features]
        y = df[self.target]

        print(f"[INFO] Feature matrix shape: {X.shape}")
        print(f"[INFO] Target vector shape : {y.shape}")
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Split data into train and test sets.
        """
        print("[INFO] Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        print(f"[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"[INFO] y_train: {y_train.shape}, y_test: {y_test.shape}")

        return X_train, X_test, y_train, y_test

    def save_features(self, X: pd.DataFrame, y: pd.Series, output_dir: str = "data/interim"):
        """
        Save combined features and target to a single CSV file for DVC tracking.
        """
        os.makedirs(output_dir, exist_ok=True)
        feature_path = os.path.join(output_dir, "features.csv")

        df_out = X.copy()
        df_out[self.target] = y
        df_out.to_csv(feature_path, index=False)

        print(f"[INFO] Saved feature dataset to: {feature_path}")

    def run(self, df: pd.DataFrame, split: bool = True):
        """
        Full feature engineering pipeline:
        - Selects features
        - Splits data
        - Saves combined features dataset (for DVC)
        """
        X, y = self.select_features(df)

        # Save all features (for DVC versioning)
        self.save_features(X, y)

        if split:
            return self.split_data(X, y)
        else:
            return X, y
