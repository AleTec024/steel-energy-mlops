import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from .config import MODEL_CONFIG, TRAINING_CONFIG

class ModelTrainer:
    """
    Trains, evaluates, and validates a Random Forest regression model.
    """

    def __init__(self, model_params=None, training_params=None):
        self.model_params = model_params or MODEL_CONFIG
        self.training_params = training_params or TRAINING_CONFIG
        self.model = RandomForestRegressor(**self.model_params)

    def train(self, X_train, y_train):
        print("[INFO] Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("[INFO] Training complete.")
        return self.model

    def evaluate(self, X_train, X_test, y_train, y_test):
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
        print("[INFO] Running cross-validation...")
        scores = cross_val_score(self.model, X, y, scoring="r2",
                                 cv=self.training_params.get("cv_folds", 5))
        print(f"[INFO] CV R² mean: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores

    def save_model(self, output_dir="models/current"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "model.pkl")
        joblib.dump(self.model, path)
        print(f"[INFO] Saved trained model to: {path}")

    def run(self, X_train, X_test, y_train, y_test):
        self.train(X_train, y_train)
        metrics = self.evaluate(X_train, X_test, y_train, y_test)
        self.save_model()
        return metrics
