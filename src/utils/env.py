# src/utils/env.py
from dotenv import load_dotenv
from pathlib import Path
import os

def load_env():
    """
    Carga variables de entorno desde el archivo .env (si existe)
    y devuelve un diccionario con las variables principales.
    """
    dotenv_path = Path(".") / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        print("✅ Archivo .env cargado correctamente.")
    else:
        print("⚠️ No se encontró .env, usando variables del entorno del sistema.")

    return {
        "ENV": os.getenv("ENV", "local"),
        "EXPERIMENT_NAME": os.getenv("EXPERIMENT_NAME", "steel-energy"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        "BACKEND_URI": os.getenv("BACKEND_URI"),
        "ARTIFACTS_URI": os.getenv("ARTIFACTS_URI"),
        "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
        "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
    }
