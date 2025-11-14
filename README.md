# Steel Industry Energy Consumption

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

ML pipeline for predicting steel industry energy consumption

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         steel_industry_energy_consumption and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── steel_industry_energy_consumption   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes steel_industry_energy_consumption a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## 🧭 Descripción General

Este proyecto implementa buenas prácticas de **MLOps** para garantizar que cualquier persona pueda **replicar los resultados**:
- **DVC**: versiona datasets y modelos.
- **MLflow**: registra parámetros, métricas y artefactos de experimentos.
- **Git LFS**: almacena binarios grandes (.pkl, .h5) cuando corresponde.
- **Pipeline**: limpieza de datos → entrenamiento (Linear, RF, XGBoost) → evaluación → logging.

---

## ✅ Requisitos

- **Python** ≥ 3.10  
- **pip**, **git**, **git-lfs**  
- **DVC** ≥ 3.x  
- **MLflow** ≥ 2.x

---

## 🚀 Instalación Rápida

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install dvc[all] mlflow
git lfs install
```

---

## ⚡️ Configuración Inicial

### 1) Clonar el repositorio

```bash
git clone https://github.com/AleTec024/steel-energy-mlops.git
cd steel-energy-mlops
git lfs pull
```

### 2) Crear y editar `.env`

```bash
cp .env.example .env
```

### 3) Recuperar datasets y modelos versionados (DVC)

```bash
dvc pull
```

---

## 🧠 Ejecución del Pipeline

```bash
dvc repro
```

---

## 📈 Seguimiento de Experimentos (MLflow)

```bash
mlflow ui --host 0.0.0.0 --port 5001
```

---

## 🔁 Reproducibilidad

```bash
git clone https://github.com/AleTec024/steel-energy-mlops.git
cd steel-energy-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install dvc[all] mlflow
git lfs install
git lfs pull
cp .env.example .env
dvc pull
dvc repro
mlflow ui --host 0.0.0.0 --port 5001
```

---

## Levantar la API

```bash
cd steel-energy-mlops
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
cd api
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Ruta y versión del artefacto del modelo (MLflow Model Registry)

Este proyecto utiliza **MLflow Model Registry** para gestionar y versionar los modelos entrenados.  
La API consume directamente la versión marcada como **`Production`**, lo que permite actualizar el modelo sin modificar la API.

### Modelos registrados

| Alias | Nombre MLflow | URI | Stage |
|------|----------------|------|--------|
| rf | steel_energy_random_forest | models:/steel_energy_random_forest/Production | Production |
| linear | steel_energy_linear | models:/steel_energy_linear/Production | Production |
| xgb | steel_energy_xgboost | models:/steel_energy_xgboost/Production | Production |

---

## Schema de entrada y salida del endpoint `/predict`

### Request — POST /predict

```json
{
  "values": [0.12, 34.5, 1.0, 540.0, 1]
}
```

### Response — PredictResponse

```json
{
  "prediction": 1234.56,
  "model_name": "rf",
  "model_source": "mlflow",
  "model_ref": "models:/steel_energy_random_forest/Production",
  "model_version": "1"
}
```

---

## Actualizar el modelo en producción (sin modificar la API)

### Paso 1 — Entrenar una nueva versión

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
dvc repro -f train_suite
```

### Paso 2 — Promover a Production

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="steel_energy_random_forest",
    version=3,
    stage="Production",
    archive_existing_versions=True
)
```
