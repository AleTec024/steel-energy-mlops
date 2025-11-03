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

--------

### 🚀 Instrucciones para ejecutar notebooks
Antes de correr cualquier notebook:

1. Asegúrate de tener configurado DVC:
   ```bash
   pip install -r requirements.txt
   dvc pull


## 🧭 Descripción General  

Este proyecto sigue las mejores prácticas de **Machine Learning Operations (MLOps)** para garantizar la **reproducibilidad total de los experimentos**.  
Incluye control de versiones de código, datos, modelos y experimentos, con una integración completa entre DVC y MLflow.

🔹 **DVC** → Versiona y rastrea datasets y modelos.  
🔹 **MLflow** → Registra experimentos, métricas y parámetros.  
🔹 **Git LFS** → Maneja artefactos grandes (.pkl, .h5).  
🔹 **Pipeline modular** → Preprocesamiento, entrenamiento, evaluación y registro automático de resultados.

Cualquier persona puede **replicar los resultados** desde cero siguiendo este README.



---

## ⚙️ Requisitos e Instalación  

### 🧰 Dependencias principales  
- Python ≥ 3.10  
- MLflow ≥ 2.x  
- DVC ≥ 3.x  
- scikit-learn, pandas, numpy, joblib  

### 🚀 Instalación rápida  

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# Instalar dependencias
pip install -r requirements.txt

# Instalar DVC y Git LFS
pip install dvc[all] mlflow
git lfs install



