import os
import pandas as pd
from sklearn.model_selection import train_test_split


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
