from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from .schemas import PredictRequest, PredictResponse
from .loader import get_model, default_model_name, vectorize
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Steel Energy – Serving API",
    description=(
        "Servicio de inferencia para **tres modelos** (linear, rf, xgb).\n\n"
        "Selecciona el modelo con el query param `?model=linear|rf|xgb`.\n"
        "Entrada: `values` (vector ordenado) **o** `features` (dict col:valor).\n"
        "Salida incluye referencia exacta del artefacto (URI/Path)."
    ),
    version="1.0.0",
)

@app.get("/health", tags=["meta"])
def health(model: Optional[str] = Query(None, description="Modelo a verificar (linear|rf|xgb).")):
    """Endpoint de estado. Permite verificar que el modelo se carga correctamente."""
    name = model or default_model_name()
    mdl, meta = get_model(name)
    return {"status": "ok", "model": name, "model_source": meta["source"], "model_ref": meta["ref"], "model_version": meta["version"]}

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest, model: Optional[str] = Query(None, description="Modelo a usar (linear|rf|xgb).")):
    """Devuelve la predicción para un payload dado."""
    try:
        name = model or default_model_name()
        mdl, meta = get_model(name)
        X = vectorize(req.features, req.values)
        y_hat = mdl.predict(X)
        pred = float(y_hat[0]) if hasattr(y_hat, "__len__") else float(y_hat)
        return PredictResponse(
            prediction=pred,
            model_name=name,
            model_source=meta["source"],
            model_ref=meta["ref"],
            model_version=meta["version"],
        )
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Falta feature requerida: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# DRIFT MONITORING ENDPOINTS
# ============================================================================

@app.post("/monitor/drift", tags=["monitoring"])
def analyze_drift(
    current_data_csv: str = Query(..., description="Path al CSV con datos actuales"),
    reference_data_csv: str = Query(
        default="../../data/raw/steel_energy_modified.csv",
        description="Path al CSV de referencia (training data)"
    ),
    save_report: bool = Query(default=True, description="Guardar reporte HTML"),
):
    """
    Analiza drift entre datos de referencia y datos actuales.

    Retorna:
        - dataset_drift_detected: bool indicando si hay drift
        - number_of_drifted_columns: cantidad de columnas con drift
        - share_of_drifted_columns: porcentaje de columnas con drift
        - drift_by_columns: detalle por columna
    """
    try:
        from .monitoring import create_monitor_from_csv

        # Cargar datos
        reference_df = pd.read_csv(reference_data_csv)
        current_df = pd.read_csv(current_data_csv)

        # Crear monitor
        monitor = create_monitor_from_csv(reference_data_csv)

        # Generar reporte
        report_path = None
        if save_report:
            report_path = "reports/drift/data_drift_report.html"

        drift_summary = monitor.generate_data_drift_report(
            current_data=current_df,
            save_path=report_path
        )

        return drift_summary

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar drift: {str(e)}")


@app.post("/monitor/performance", tags=["monitoring"])
def analyze_performance(
    current_data_csv: str = Query(..., description="Path al CSV con predicciones y targets reales"),
    reference_data_csv: str = Query(
        default="../../data/raw/steel_energy_modified.csv",
        description="Path al CSV de referencia"
    ),
    save_report: bool = Query(default=True, description="Guardar reporte HTML"),
):
    """
    Analiza performance del modelo comparando predicciones vs targets reales.

    El CSV actual debe contener columnas 'prediction' y 'Usage_kWh' (target).

    Retorna métricas: MAE, MAPE, RMSE, R2
    """
    try:
        from .monitoring import create_monitor_from_csv

        # Crear monitor
        monitor = create_monitor_from_csv(reference_data_csv)

        # Cargar datos actuales
        current_df = pd.read_csv(current_data_csv)

        # Verificar que existan columnas necesarias
        if "prediction" not in current_df.columns or "Usage_kWh" not in current_df.columns:
            raise ValueError("El CSV actual debe contener columnas 'prediction' y 'Usage_kWh'")

        # Generar reporte
        report_path = None
        if save_report:
            report_path = "reports/drift/performance_report.html"

        performance_summary = monitor.generate_model_performance_report(
            current_data=current_df,
            save_path=report_path
        )

        return performance_summary

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar performance: {str(e)}")


@app.get("/monitor/reports/{report_name}", tags=["monitoring"])
def get_report(report_name: str):
    """
    Descarga un reporte HTML generado previamente.

    Ejemplos:
        - /monitor/reports/data_drift_report.html
        - /monitor/reports/performance_report.html
    """
    report_path = Path(f"reports/drift/{report_name}")

    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Reporte no encontrado: {report_name}")

    return FileResponse(
        path=str(report_path),
        media_type="text/html",
        filename=report_name
    )


@app.post("/monitor/simulate-drift", tags=["monitoring"])
def simulate_drift_scenario(
    scenario: str = Query(
        default="moderate_drift",
        description="Escenario de drift: no_drift, mild_drift, moderate_drift, severe_drift"
    ),
    n_samples: int = Query(default=200, description="Número de muestras a generar"),
    reference_data_csv: str = Query(
        default="../../data/raw/steel_energy_modified.csv",
        description="Path al CSV de referencia"
    ),
):
    """
    Genera datos sintéticos con drift simulado para testing.

    Retorna el dataset generado y análisis de drift.
    """
    try:
        from .data_simulator import DriftSimulator
        from .monitoring import create_monitor_from_csv

        # Cargar datos de referencia
        reference_df = pd.read_csv(reference_data_csv)

        # Crear simulador
        simulator = DriftSimulator(reference_df)

        # Generar datos con drift
        drifted_data = simulator.create_drift_scenario(scenario, n_samples=n_samples)

        # Guardar datos generados
        output_path = f"reports/drift/simulated_{scenario}_{n_samples}.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        drifted_data.to_csv(output_path, index=False)

        # Analizar drift
        monitor = create_monitor_from_csv(reference_data_csv)
        drift_summary = monitor.generate_data_drift_report(
            current_data=drifted_data,
            save_path=f"reports/drift/simulated_{scenario}_drift_report.html"
        )

        return {
            "scenario": scenario,
            "n_samples": n_samples,
            "simulated_data_path": output_path,
            "drift_analysis": drift_summary,
            "report_url": f"/monitor/reports/simulated_{scenario}_drift_report.html"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al simular drift: {str(e)}")
