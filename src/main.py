# src/main.py
import os
import argparse
import yaml
import pandas as pd

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer

# Import trainers (lightweight registry)
from src.models.xgboost_model import ModelTrainer as XGBTrainer
from src.models.random_forest_model import ModelTrainer as RFTrainer
from src.models.linear_regression_model import ModelTrainer as LRTrainer

MODEL_REGISTRY = {
    "xgboost": XGBTrainer,
    "random_forest": RFTrainer,
    "linear_regression": LRTrainer,
}

def load_cfg(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_data_loader(cfg):
    raw = cfg["data"]["raw_path"]
    processed = cfg["data"]["processed_path"]
    print("=" * 70); print("[INFO] STEP 1: Running DataLoader"); print("=" * 70)
    DataLoader(raw, processed).run()
    print(f"[INFO] Processed file at: {processed}")
    return processed

def run_feature_engineering(cfg, processed_path):
    print("=" * 70); print("[INFO] STEP 2: Running Feature Engineering"); print("=" * 70)
    df = pd.read_csv(processed_path)
    features = cfg["features"]["selected"]
    target = cfg["features"]["target"]
    X_train, X_test, y_train, y_test = FeatureEngineer(features, target).run(df)
    print(f"[INFO] X_train {X_train.shape} | X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def run_training(cfg, X_train, X_test, y_train, y_test):
    print("=" * 70); print("[INFO] STEP 3: Training Model"); print("=" * 70)
    model_type = cfg["train"]["model_type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type '{model_type}'. "
                         f"Use one of: {list(MODEL_REGISTRY.keys())}")

    # Pick params for the selected model (xgboost / random_forest / linear_regression)
    model_params = cfg["train"].get(model_type, {})
    trainer_cls = MODEL_REGISTRY[model_type]
    trainer = trainer_cls(model_params=model_params)

    metrics = trainer.run(X_train, X_test, y_train, y_test)
    print("=" * 70); print("[INFO] STEP 4: Cross-Validation"); print("=" * 70)
    trainer.cross_validate(X_train, y_train)

    print("=" * 70); print("[INFO] STEP 5: Pipeline Summary"); print("=" * 70)
    print(f"Model Type: {model_type}")
    print(f"Metrics: {metrics}")
    print("\n[INFO] ✅ Full pipeline executed successfully!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "data_loader", "feature_engineering", "train"])
    args = parser.parse_args()

    cfg = load_cfg()  # reads params.yaml

    # Execute data_loader stage if needed
    if args.stage in ("all", "data_loader"):
        processed = run_data_loader(cfg)
    else:
        processed = cfg["data"]["processed_path"]

    # Execute feature_engineering stage if needed
    if args.stage in ("all", "feature_engineering"):
        X_train, X_test, y_train, y_test = run_feature_engineering(cfg, processed)

    # Execute train stage if needed
    if args.stage == "train":
        # If someone runs "train" directly, load features.csv (created by FeatureEngineer)
        df = pd.read_csv("data/interim/features.csv")
        features = cfg["features"]["selected"]; target = cfg["features"]["target"]
        X = df[features]; y = df[target]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        run_training(cfg, X_train, X_test, y_train, y_test)
    elif args.stage == "all":
        # If running full pipeline, X_train/X_test already exist from feature_engineering
        run_training(cfg, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()