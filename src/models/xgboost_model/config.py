# src/models/xgboost_model/config.py

MODEL_CONFIG = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1
}

TRAINING_CONFIG = {
    "cv_folds": 5,
    "test_size": 0.2,
}
