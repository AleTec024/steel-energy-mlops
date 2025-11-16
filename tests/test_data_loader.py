from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from data.data_loader import DataLoader


@pytest.fixture
def raw_dataframe():
    return pd.DataFrame(
        {
            "date": ["2024-01-01 05:00:00", "not-a-date", "2024-02-15 18:30:00"],
            "lagging_current_reactive.power_kvarh": [1.0, 2.0, 3.0],
            "leading_current_reactive_power_kvarh": [0.1, 0.2, 0.3],
            "lagging_current_power_factor": [0.9, 0.8, 0.95],
            "leading_current_power_factor": [0.1, 0.2, 0.05],
            "nsm": [100, 200, 300],
            "weekstatus": ["Weekday", "Weekday", "Weekend"],
            "day_of_week": ["Monday", "Tuesday", "Sunday"],
            "load_type": ["heavy", "heavy", "light"],
            "usage_kwh": [10, 15, 30],
        }
    )


def test_load_data_missing_file(tmp_path):
    loader = DataLoader(str(tmp_path / "missing.csv"))
    with pytest.raises(FileNotFoundError):
        loader.load_data()


def test_load_data_missing_required_columns(tmp_path, raw_dataframe):
    bad_path = tmp_path / "incomplete.csv"
    raw_dataframe.drop(columns=["weekstatus"]).to_csv(bad_path, index=False)

    loader = DataLoader(str(bad_path))
    with pytest.raises(ValueError):
        loader.load_data()


def test_preprocess_adds_time_and_dummy_features(raw_dataframe):
    loader = DataLoader("unused.csv")

    processed = loader.preprocess(raw_dataframe)

    # One invalid date row should be dropped
    assert processed.shape[0] == 2
    assert processed["date"].is_monotonic_increasing

    time_cols = {"hour", "dayofweek_num", "month"}
    dummy_cols = {"weekstatus_Weekend", "day_of_week_Sunday", "load_type_light"}
    assert time_cols.issubset(processed.columns)
    assert dummy_cols.issubset(processed.columns)

    # Confirm time feature extraction
    assert processed.iloc[0]["hour"] == 5
    assert processed.iloc[1]["hour"] == 18
    assert processed.iloc[0]["dayofweek_num"] == 0  # Monday
    assert processed.iloc[1]["dayofweek_num"] == 3  # Thursday


def test_run_processes_and_persists_processed_csv(tmp_path, raw_dataframe):
    input_path = tmp_path / "raw.csv"
    raw_dataframe.to_csv(input_path, index=False)

    loader = DataLoader(str(input_path))
    processed = loader.run()

    output_path = Path(loader.output_path)
    assert output_path.exists()

    saved = pd.read_csv(output_path)
    saved["date"] = pd.to_datetime(saved["date"])
    assert_frame_equal(saved.reset_index(drop=True), processed.reset_index(drop=True), check_dtype=False)
