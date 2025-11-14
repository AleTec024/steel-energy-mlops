import os, json

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def load_catalog():
    path = os.path.join(BASE_DIR, "model_catalog.json")
    with open(path, "r") as f:
        return json.load(f)

def load_feature_order():
    path = os.path.join(BASE_DIR, "feature_config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg.get("feature_order")
