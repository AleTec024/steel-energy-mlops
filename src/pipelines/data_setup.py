"""Utilities to load the cleaned dataset and prepare feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data_loader import DataLoader

DEFAULT_DATA_REL_PATH = Path("data/clean/steel_energy_cleaned_v2.csv")

DATE_FEATURE_NAMES = [
    "date_hour",
    "date_dayofweek",
    "date_month",
    "date_dayofyear",
    "date_sin_hour",
    "date_cos_hour",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Captures the column selections used across experiments."""

    categorical_features: list[str]
    numeric_base_features: list[str]
    interaction_columns: list[str]
    target: str = "usage_kwh"

    @property
    def date_feature_names(self) -> list[str]:
        return DATE_FEATURE_NAMES

    @property
    def interaction_feature_names(self) -> list[str]:
        return [f"{a}__div__{b}" for a, b in combinations(self.interaction_columns, 2)]

    @property
    def numeric_features_for_scaling(self) -> list[str]:
        return (
            self.numeric_base_features
            + self.date_feature_names
            + self.interaction_feature_names
        )

    def to_dict(self) -> Dict[str, list[str]]:
        return {
            "categorical_features": self.categorical_features,
            "numeric_base_features": self.numeric_base_features,
            "date_feature_names": self.date_feature_names,
            "interaction_columns": self.interaction_columns,
            "interaction_feature_names": self.interaction_feature_names,
            "numeric_features_for_scaling": self.numeric_features_for_scaling,
            "target": [self.target],
        }


DEFAULT_FEATURE_CONFIG = FeatureConfig(
    categorical_features=["weekstatus", "day_of_week", "load_type"],
    numeric_base_features=[
        "lagging_current_reactive.power_kvarh",
        "leading_current_reactive_power_kvarh",
        "co2(tco2)",
        "lagging_current_power_factor",
        "leading_current_power_factor",
        "nsm",
    ],
    interaction_columns=[
        "lagging_current_reactive.power_kvarh",
        "leading_current_power_factor",
    ],
)


def infer_project_root(start: Optional[Path] = None) -> Path:
    """Walk upwards until we find the repository root."""
    search_path = start or Path.cwd()
    for candidate in [search_path, *search_path.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not infer project root (missing data/ or src/).")


def resolve_data_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or infer_project_root()
    path = root / DEFAULT_DATA_REL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Clean dataset not found at {path}")
    return path


def load_clean_dataframe(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load, parse dates, and clean the prepared CSV."""
    path = data_path or resolve_data_path()
    loader = DataLoader(str(path))
    df = loader.load_data()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def build_feature_frame(
    df: pd.DataFrame, config: FeatureConfig = DEFAULT_FEATURE_CONFIG
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target series based on the configuration."""
    required_columns = (
        config.categorical_features
        + config.numeric_base_features
        + ["date", config.target]
    )

    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    feature_df = df[config.categorical_features + config.numeric_base_features + ["date"]]
    target = df[config.target]
    return feature_df, target


def split_train_test(
    feature_df: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Wrapper around train_test_split with project defaults."""
    return train_test_split(
        feature_df,
        target,
        test_size=test_size,
        random_state=random_state,
    )

