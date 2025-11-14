import os
import numpy as np
from .catalog import load_catalog, load_feature_order

os.environ.setdefault("PYTHONHASHSEED", "0")

CATALOG = load_catalog()
FEATURE_ORDER = load_feature_order()
_CACHE = {}  # cache de modelos por nombre

def _load_mlflow(uri: str):
    import mlflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(uri)
    version = None
    parts = uri.split("/")
    if len(parts) >= 3 and parts[-1].isdigit():
        version = parts[-1]
    return model, {"source": "mlflow", "ref": uri, "version": version}

def _load_local(path: str):
    import joblib
    model = joblib.load(path)
    return model, {"source": "local", "ref": path, "version": None}

def get_model(model_name: str):
    if model_name in _CACHE:
        return _CACHE[model_name]

    spec = CATALOG["models"].get(model_name)
    if not spec:
        raise ValueError(f"Modelo '{model_name}' no está definido en model_catalog.json")

    if spec["mode"] == "mlflow":
        model, meta = _load_mlflow(spec["uri"])
    else:
        model, meta = _load_local(spec["path"])

    _CACHE[model_name] = (model, meta)
    return _CACHE[model_name]

def default_model_name():
    return CATALOG.get("default_model", list(CATALOG["models"].keys())[0])

def vectorize(payload_features, payload_values):
    if payload_values is not None:
        return np.array(payload_values, dtype=float).reshape(1, -1)
    if payload_features is None:
        raise ValueError("Debes mandar 'values' o 'features'.")
    if FEATURE_ORDER:
        arr = [float(payload_features[name]) for name in FEATURE_ORDER]
    else:
        arr = [float(v) for _, v in payload_features.items()]
    return np.array(arr, dtype=float).reshape(1, -1)
