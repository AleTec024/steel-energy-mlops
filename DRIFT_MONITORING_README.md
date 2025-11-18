# 🔍 Drift Monitoring con Evidently AI

Sistema de monitoreo de drift para el proyecto Steel Energy MLOps. Detecta cambios en distribuciones de datos y degradación del modelo en producción.

## ⚡ Quick Info

**Dataset usado**: `data/clean/steel_energy_cleaned_v2.csv` (nombres de columnas en minúsculas)

**Reportes generados**:
- 📊 **Drift en features** (X): Análisis de distribuciones de variables
- 📈 **Performance del modelo**: Gráficas **Predicción vs Real** + métricas (MAE, RMSE, R²)
- 📉 **Reportes comprehensivos**: Drift + Performance en un solo reporte

**¿Quieres ver gráficas de Predicción vs Real?**
- Abre: `demo_comprehensive_report.html` o `demo_concept_drift_performance.html`

## 📋 Tabla de Contenidos

- [¿Qué es Drift?](#qué-es-drift)
- [Instalación](#instalación)
- [Quick Start](#quick-start)
- [Uso de la API](#uso-de-la-api)
- [Tipos de Drift](#tipos-de-drift)
- [Ejemplos](#ejemplos)

---

## ¿Qué es Drift?

El **drift** ocurre cuando los datos en producción cambian respecto a los datos de entrenamiento, causando degradación del modelo.

### Tipos de Drift

1. **Data Drift (Covariate Shift)**: Cambios en distribución de features (X)
   - Ejemplo: Cambio de estacionalidad, nuevas condiciones operacionales

2. **Concept Drift**: Cambios en relación feature-target (X → y)
   - Ejemplo: El modelo deja de predecir bien aunque los datos luzcan similares

3. **Sudden Drift**: Cambios abruptos
   - Ejemplo: Cambio de maquinaria, nuevos procesos

4. **Gradual Drift**: Cambios lentos en el tiempo
   - Ejemplo: Desgaste de equipos, evolución de procesos

---

## Instalación

### 1. Instalar dependencias

```bash
pip install -r api/requirements.txt
```

La dependencia clave agregada es:
```
evidently==0.7.14
```

### 2. Verificar instalación

```python
import evidently
print(evidently.__version__)  # Debe mostrar 0.7.14
```

---

## Quick Start

### Demo en 3 pasos

```bash
# 1. Ejecutar script de demostración
python demo_drift_monitoring.py

# 2. Abrir reportes HTML generados
# Los reportes se guardan en: reports/drift/*.html

# 3. Ver en navegador (ejemplo)
start reports/drift/demo_comprehensive_report.html  # Windows
open reports/drift/demo_comprehensive_report.html   # Mac
xdg-open reports/drift/demo_comprehensive_report.html  # Linux
```

Esto genera **10 reportes HTML**:
- ✅ Reportes HTML interactivos con gráficas
- ✅ Análisis de drift para diferentes escenarios (no_drift, mild, moderate, severe)
- ✅ Métricas de performance (MAE, RMSE, R²) con gráficas **Predicción vs Real**
- ✅ Reportes de drift abrupto (high_load, low_load, anomaly)
- ✅ Métricas JSON para automatización

### 📊 Tipos de Reportes Generados

**Reportes de DRIFT (solo análisis de features):**
- `demo_no_drift_report.html` - Sin cambios en distribuciones
- `demo_mild_drift_report.html` - Cambios leves (20%)
- `demo_moderate_drift_report.html` - Cambios medios (50%)
- `demo_severe_drift_report.html` - Cambios severos (80%)

**Reportes de PERFORMANCE (con gráficas Predicción vs Real):**
- `demo_concept_drift_performance.html` - ⭐ **Scatter plot: Predicción vs Real**
- `demo_comprehensive_report.html` - ⭐ **Drift + Performance + Predicción vs Real**

**Reportes de SUDDEN DRIFT:**
- `demo_sudden_high_load.html` - Simulación alta carga
- `demo_sudden_low_load.html` - Simulación baja carga
- `demo_sudden_anomaly.html` - Valores anómalos

### 🎯 ¿Qué reporte abrir?

- **Para ver Predicción vs Real**: `demo_comprehensive_report.html` o `demo_concept_drift_performance.html`
- **Para ver drift en features**: Cualquier `demo_*_drift_report.html`
- **Para análisis completo**: `demo_comprehensive_report.html` (tiene todo)

---

## Uso de la API

### 1. Iniciar el servidor

```bash
cd api
uvicorn app.main:app --reload --port 8000
```

### 2. Acceder a la documentación

Abre en tu navegador: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Endpoints disponibles

#### 🔹 Simular Drift y Generar Reporte

```bash
curl -X POST "http://localhost:8000/monitor/simulate-drift?scenario=severe_drift&n_samples=300" \
  -H "accept: application/json"
```

**Parámetros:**
- `scenario`: `no_drift` | `mild_drift` | `moderate_drift` | `severe_drift`
- `n_samples`: Número de muestras a generar (default: 200)
- `reference_data_csv`: Path al CSV de referencia (default: datos de entrenamiento)

**Respuesta:**
```json
{
  "scenario": "severe_drift",
  "n_samples": 300,
  "simulated_data_path": "reports/drift/simulated_severe_drift_300.csv",
  "drift_analysis": {
    "dataset_drift_detected": true,
    "number_of_drifted_columns": 8,
    "share_of_drifted_columns": 0.73,
    "timestamp": "2025-01-17T10:30:45"
  },
  "report_url": "/monitor/reports/simulated_severe_drift_drift_report.html"
}
```

#### 🔹 Analizar Drift en Datos Propios

```bash
curl -X POST "http://localhost:8000/monitor/drift?current_data_csv=data/production_data.csv" \
  -H "accept: application/json"
```

**Parámetros:**
- `current_data_csv`: Path al CSV con datos de producción (requerido)
- `reference_data_csv`: Path al CSV de entrenamiento (opcional)
- `save_report`: Guardar HTML (default: true)

**Respuesta:**
```json
{
  "dataset_drift_detected": true,
  "number_of_drifted_columns": 5,
  "share_of_drifted_columns": 0.45,
  "drift_by_columns": {
    "Lagging_Current_Reactive.Power_kVarh": {
      "drift_detected": true,
      "drift_score": 0.23,
      "stattest_name": "ks"
    },
    "NSM": {
      "drift_detected": false,
      "drift_score": 0.05,
      "stattest_name": "ks"
    }
  },
  "timestamp": "2025-01-17T10:35:22"
}
```

#### 🔹 Analizar Performance del Modelo

```bash
curl -X POST "http://localhost:8000/monitor/performance?current_data_csv=data/predictions_with_actuals.csv" \
  -H "accept: application/json"
```

**Requisitos:**
- El CSV debe contener:
  - Columna `prediction`: Predicciones del modelo
  - Columna `usage_kwh`: Valores reales (target)

**Nota**: El dataset limpio usa nombres en **minúsculas**: `usage_kwh`, `lagging_current_reactive.power_kvarh`, `nsm`, `co2(tco2)`, etc.

**Respuesta:**
```json
{
  "mae": 45.23,
  "mape": 12.5,
  "rmse": 67.89,
  "r2_score": 0.85,
  "timestamp": "2025-01-17T10:40:15"
}
```

#### 🔹 Descargar Reporte HTML

```bash
curl "http://localhost:8000/monitor/reports/data_drift_report.html" \
  --output drift_report.html
```

O abre directamente en navegador:
```
http://localhost:8000/monitor/reports/data_drift_report.html
```

---

## Tipos de Drift

### 1. Covariate Drift (Cambios en Features)

```python
from api.app.data_simulator import DriftSimulator
from api.app.monitoring import DriftMonitor
import pandas as pd

# Cargar datos limpios
reference_df = pd.read_csv("data/clean/steel_energy_cleaned_v2.csv")

# Simular drift
simulator = DriftSimulator(reference_df)
drifted_data = simulator.simulate_covariate_drift(
    n_samples=200,
    drift_magnitude=0.5,  # 0.0 = sin drift, 1.0 = drift extremo
    columns_to_drift=["lagging_current_reactive.power_kvarh", "nsm"]
)

# Detectar drift
monitor = DriftMonitor(reference_data=reference_df)
drift_summary = monitor.generate_data_drift_report(
    current_data=drifted_data,
    save_path="reports/drift/covariate_drift.html"
)

print(f"Drift detectado: {drift_summary['dataset_drift_detected']}")
print(f"Columnas con drift: {drift_summary['number_of_drifted_columns']}")
```

### 2. Concept Drift (Degradación del Modelo)

```python
import numpy as np

# Agregar predicciones ruidosas
reference_df["prediction"] = reference_df["usage_kwh"] + np.random.normal(0, 0.1, len(reference_df))

# Simular degradación
current_df = simulator.simulate_concept_drift(
    n_samples=200,
    noise_factor=0.8  # Más ruido = más degradación
)
current_df["prediction"] = current_df["usage_kwh"] + np.random.normal(0, 2.0, len(current_df))

# Analizar performance
monitor = DriftMonitor(
    reference_data=reference_df,
    target_column="usage_kwh",
    prediction_column="prediction"
)

performance = monitor.generate_model_performance_report(
    current_data=current_df,
    save_path="reports/drift/performance.html"
)

print(f"MAE: {performance['mae']:.2f}")
print(f"RMSE: {performance['rmse']:.2f}")
print(f"R²: {performance['r2_score']:.3f}")
```

### 3. Sudden Drift (Cambios Abruptos)

```python
# Simular escenarios extremos
drift_types = ["high_load", "low_load", "anomaly"]

for drift_type in drift_types:
    drifted_data = simulator.simulate_sudden_drift(
        n_samples=150,
        drift_type=drift_type
    )

    drift_summary = monitor.generate_data_drift_report(
        current_data=drifted_data,
        save_path=f"reports/drift/sudden_{drift_type}.html"
    )
```

### 4. Gradual Drift (Cambios Lentos)

```python
# Drift progresivo en el tiempo
drifted_data = simulator.simulate_gradual_drift(
    n_samples=500,
    drift_speed=0.5  # 0.0 = lento, 1.0 = rápido
)

drift_summary = monitor.generate_data_drift_report(
    current_data=drifted_data,
    save_path="reports/drift/gradual_drift.html"
)
```

---

## Ejemplos

### Ejemplo 1: Pipeline de Monitoreo Automático

```python
import pandas as pd
from api.app.monitoring import create_monitor_from_csv

# Configurar monitor
monitor = create_monitor_from_csv(
    reference_csv_path="data/clean/steel_energy_cleaned_v2.csv",
    target_column="usage_kwh",
    prediction_column="prediction"
)

# Cargar datos de producción
production_data = pd.read_csv("data/production/latest_batch.csv")

# Analizar drift
drift_report = monitor.generate_data_drift_report(
    current_data=production_data,
    save_path="reports/drift/production_drift.html"
)

# Alertas automáticas
if drift_report["dataset_drift_detected"]:
    print("⚠️  ALERTA: Drift detectado en producción!")
    print(f"   Columnas afectadas: {drift_report['number_of_drifted_columns']}")

    # Enviar notificación, reentrenar modelo, etc.
    # send_slack_alert(drift_report)
else:
    print("✅ Sin drift detectado. Modelo operando normalmente.")
```

### Ejemplo 2: Comparar Múltiples Períodos

```python
from datetime import datetime
import glob

# Analizar drift por semana
weekly_files = glob.glob("data/production/week_*.csv")

drift_history = []

for file in weekly_files:
    week_data = pd.read_csv(file)

    drift_summary = monitor.generate_data_drift_report(
        current_data=week_data,
        save_path=f"reports/drift/{Path(file).stem}_drift.html"
    )

    drift_history.append({
        "week": Path(file).stem,
        "drift_detected": drift_summary["dataset_drift_detected"],
        "drifted_columns": drift_summary["number_of_drifted_columns"],
        "timestamp": datetime.now()
    })

# Convertir a DataFrame para análisis
drift_df = pd.DataFrame(drift_history)
print(drift_df)
```

### Ejemplo 3: Integración con MLflow

```python
import mlflow

# Iniciar experimento
mlflow.start_run()

# Generar reporte
drift_summary = monitor.generate_data_drift_report(
    current_data=production_data,
    save_path="reports/drift/current_drift.html"
)

# Loggear métricas en MLflow
mlflow.log_metric("dataset_drift", int(drift_summary["dataset_drift_detected"]))
mlflow.log_metric("drifted_columns_count", drift_summary["number_of_drifted_columns"])
mlflow.log_metric("drifted_columns_share", drift_summary["share_of_drifted_columns"])
mlflow.log_artifact("reports/drift/current_drift.html")

mlflow.end_run()
```

---

## 📊 Interpretación de Reportes

### Reporte HTML de Evidently

Los reportes incluyen:

1. **Dataset Drift Summary**
   - ¿Hay drift general? (True/False)
   - Número y porcentaje de features con drift
   - Score global de drift

2. **Drift por Columna**
   - Test estadístico usado (KS test, Chi-squared, etc.)
   - P-value y drift score
   - Distribuciones antes/después (histogramas)

3. **Regression Performance** (si hay target)
   - MAE, RMSE, MAPE, R²
   - Scatter plot: Predicciones vs Actuals
   - Residual plots

### Umbrales de Alerta

```python
# Configurar umbrales personalizados
if drift_summary["share_of_drifted_columns"] > 0.3:
    print("🔴 CRÍTICO: Más del 30% de features con drift")
elif drift_summary["share_of_drifted_columns"] > 0.15:
    print("🟡 ADVERTENCIA: 15-30% de features con drift")
else:
    print("🟢 OK: Drift mínimo o nulo")
```

---

## 🚀 Mejores Prácticas

1. **Definir datos de referencia claros**
   - Usar datos de entrenamiento o validación
   - Asegurar que sean representativos

2. **Monitorear regularmente**
   - Configurar alertas automáticas
   - Revisar reportes semanalmente

3. **Actuar sobre drift detectado**
   - Re-entrenar modelo con datos recientes
   - Investigar causas raíz
   - Actualizar features si es necesario

4. **Versionar reportes**
   - Guardar reportes con timestamp
   - Comparar evolución temporal

---

## 🐛 Troubleshooting

### Error: "Column not found"

```python
# Verificar columnas disponibles
print(production_data.columns.tolist())

# Asegurar que coincidan con reference data
print(reference_data.columns.tolist())
```

### Error: "No drift metrics"

```python
# Asegurar que hay suficientes datos
print(f"Reference size: {len(reference_data)}")
print(f"Current size: {len(current_data)}")

# Mínimo recomendado: 100+ muestras
```

### Reportes no se generan

```python
# Verificar permisos de escritura
from pathlib import Path
Path("reports/drift").mkdir(parents=True, exist_ok=True)
```

---

## 📚 Recursos Adicionales

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Drift Detection Tutorial](https://www.evidentlyai.com/blog/tutorial-1-model-analytics-in-production)
- [MLOps Best Practices](https://ml-ops.org/)

---

## ✅ Checklist de Implementación

- [x] Módulo de monitoreo (`api/app/monitoring.py`)
- [x] Simulador de drift (`api/app/data_simulator.py`)
- [x] Endpoints en API (`/monitor/*`)
- [x] Script de demostración (`demo_drift_monitoring.py`)
- [x] Dependencia Evidently agregada (`api/requirements.txt`)
- [ ] Configurar alertas automáticas (Slack, email, etc.)
- [ ] Integrar con MLflow tracking
- [ ] Scheduler para monitoreo periódico (Airflow, cron, etc.)
- [ ] Dashboard de monitoreo en tiempo real

---

**¿Preguntas?** Abre un issue en el repositorio o consulta la documentación de Evidently AI.
