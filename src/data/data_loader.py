import os
import pandas as pd
from datetime import datetime

class DataLoader:
    """
    Handles loading, validation, cleaning, and preprocessing of the steel energy dataset.
    """

    def __init__(self, input_path: str, output_path: str = None):
        """
        Initialize DataLoader with input and optional output paths.
        """
        self.input_path = input_path
        self.output_path = output_path or os.path.join(
            os.path.dirname(input_path), "..", "processed", "steel_energy_processed.csv"
        )

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV and perform basic validation.
        """
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"File not found: {self.input_path}")

        df = pd.read_csv(self.input_path)
        print(f"[INFO] Loaded dataset — Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        expected_columns = [
            "date",
            "lagging_current_reactive.power_kvarh",
            "leading_current_reactive_power_kvarh",
            "lagging_current_power_factor",
            "leading_current_power_factor",
            "nsm",
            "weekstatus",
            "day_of_week",
            "load_type",
            "usage_kwh",
        ]

        missing_cols = [c for c in expected_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")

        print("[INFO] Column validation passed.")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess dataset:
        - Convert date column
        - Drop nulls
        - Sort by date
        - Add time features
        - Encode categorical variables
        """
        print("[INFO] Starting preprocessing...")

        # Convert date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Drop nulls
        initial_rows = df.shape[0]
        df = df.dropna().sort_values(by="date")
        print(f"[INFO] Dropped {initial_rows - df.shape[0]} null rows.")

        # Add time-based columns
        df["hour"] = df["date"].dt.hour
        df["dayofweek_num"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
        df["month"] = df["date"].dt.month

        # One-hot encoding
        categorical_cols = ["weekstatus", "day_of_week", "load_type"]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        print("[INFO] Preprocessing complete.")
        print(f"[INFO] Final dataset shape: {df.shape}")

        return df

    def save_processed(self, df: pd.DataFrame):
        """
        Save processed dataset to disk.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"[INFO] Processed dataset saved to: {self.output_path}")

    def run(self) -> pd.DataFrame:
        """
        Execute the full load → preprocess → save pipeline.
        """
        df = self.load_data()
        df = self.preprocess(df)
        self.save_processed(df)
        return df
