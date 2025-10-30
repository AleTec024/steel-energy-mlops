# src/models/xgboost_model/model_trainer.py
import numpy as np
import xgboost as xgb
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from .config import MODEL_CONFIG, TRAINING_CONFIG


class ModelTrainer:
    """
    Trains, evaluates, and validates an XGBoost regression model.
    """

    def __init__(self, model_params=None, training_params=None):
        self.model_params = model_params or MODEL_CONFIG
        self.training_params = training_params or TRAINING_CONFIG
        self.model = xgb.XGBRegressor(**self.model_params)

    def train(self, X_train, y_train):
        """
        Train the XGBoost model on the given data.
        """
        print("[INFO] Training XGBoost model...")
        self.model.fit(X_train, y_train)
        print("[INFO] Training complete.")
        return self.model

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        Evaluate model performance on training and test sets.
        """
        print("[INFO] Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)

        metrics = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2_test": r2_score(y_test, y_pred),
            "R2_train": r2_score(y_train, y_train_pred),
        }

        print("[INFO] Model Evaluation:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
        return metrics

    def cross_validate(self, X, y):
        """
        Perform k-fold cross-validation on the training data.
        """
        print("[INFO] Running cross-validation...")
        scores = cross_val_score(
            self.model, X, y,
            scoring="r2",
            cv=self.training_params.get("cv_folds", 5)
        )
        print(f"[INFO] CV R² mean: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores
       
    def save_model(self, output_dir="models/xgboost_model/artifacts"):
        """
        Save the trained XGBoost model to a JSON file for DVC tracking.
        """
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.json")
        self.model.save_model(model_path)
        print(f"[INFO] Saved trained model to: {model_path}")

    def run(self, X_train, X_test, y_train, y_test):
        """
        Full training + evaluation pipeline.
        """
        self.train(X_train, y_train)
        metrics = self.evaluate(X_train, X_test, y_train, y_test)
        self.save_model()
        return metrics
