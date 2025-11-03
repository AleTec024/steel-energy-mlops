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


---

## ✅ Requisitos

- **Python** ≥ 3.10  
- **pip**, **git**, **git-lfs**  
- **DVC** ≥ 3.x  
- **MLflow** ≥ 2.x

---

## 🚀 Instalación Rápida

```bash
# 1) Crear y activar entorno virtual
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Instalar herramientas adicionales
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

Configura estos valores en `.env`:

```env
# MLflow
MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# (Opcional) MLflow Server con Postgres + artefactos remotos
BACKEND_URI=postgresql://USER:PASS@HOST:5432/DBNAME
ARTIFACTS_URI=file://$(pwd)/mlruns_artifacts   # o s3://tu-bucket/prefix
```

### 3) Recuperar datasets y modelos versionados (DVC)

```bash
dvc pull
```

---

## 🧠 Ejecución del Pipeline

### Opción A — Ejecutar TODO con DVC

```bash
dvc repro
```

Esto ejecuta las etapas declaradas en `dvc.yaml`:
- Limpieza / transformación de datos → `data/clean/`
- Entrenamiento de modelos (Linear, RF, XGB)
- Evaluación y registro de métricas/artefactos en MLflow

### Opción B — Entrenar por modelo

```bash
python src/models/linear_regression_model/train.py
python src/models/random_forest_model/train.py
python src/models/xgboost_model/train.py
```

---

## 📈 Seguimiento de Experimentos (MLflow)

### UI local (rápida)

```bash
mlflow ui --host 0.0.0.0 --port 5001
```

Navega a: http://localhost:5001

### Servidor MLflow (Postgres + Artefactos remotos)

```bash
# Cargar variables .env en la shell actual
export $(grep -v '^#' .env | xargs)

mlflow server \
  --backend-store-uri "$BACKEND_URI" \
  --artifacts-destination "$ARTIFACTS_URI" \
  --host 0.0.0.0 --port 5001
```

> **Nota:** si MLflow te pide migrar esquema:  
> 1) haz backup de la base y 2) ejecuta:
>
> ```bash
> mlflow db upgrade "$BACKEND_URI"
> ```

---

## 🔁 Reproducibilidad

Pasos para replicar resultados **de principio a fin**:

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
# (editar .env si usarás servidor MLflow/artefactos remotos)
dvc pull
dvc repro
mlflow ui --host 0.0.0.0 --port 5001
```

✅ Los resultados (métricas/artefactos) deben coincidir con lo reportado en MLflow y DVC.

---

## 🧰 Herramientas y Versiones

| Herramienta     | Versión recomendada |
|-----------------|---------------------|
| Python          | 3.10+               |
| DVC             | 3.x                 |
| MLflow          | 2.x                 |
| scikit-learn    | 1.5+                |
| pandas          | 2.x                 |
| numpy           | 1.26+               |

---