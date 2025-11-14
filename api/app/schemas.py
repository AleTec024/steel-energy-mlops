# api/app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class PredictRequest(BaseModel):
    """
    Schema de entrada para /predict.
    Envía 'values' (vector ordenado) o 'features' (dict col:valor).
    """
    features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Mapa feature→valor usando nombres de columna."
    )
    # Pydantic v2: usa List[float] + min_length
    values: Optional[List[float]] = Field(
        default=None,
        min_length=1,
        description="Vector ordenado según feature_config.json."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"values": [0.12, 34.5, 1.0, 7.8, 0.0]},
                {"features": {
                    "Lagging_Current_reactive_power_kVarh": 12.3,
                    "Leading_Current_reactive_power_kVarh": 0.0,
                    "CO2(t)": 1.7,
                    "NSM": 540,
                    "WeekStatus": 1
                }}
            ]
        }
    }

class PredictResponse(BaseModel):
    """Schema de salida de /predict."""
    prediction: float = Field(description="Predicción del modelo.")
    model_name: str = Field(description="Identificador del modelo usado (linear|rf|xgb).")
    model_source: str = Field(description="Origen del artefacto: mlflow|local.")
    model_ref: str = Field(description="URI (MLflow) o PATH (local).")
    model_version: Optional[str] = Field(default=None, description="Versión si aplica (MLflow).")
