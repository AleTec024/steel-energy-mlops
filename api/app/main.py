from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
from .schemas import PredictRequest, PredictResponse
from .loader import get_model, default_model_name, vectorize

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
