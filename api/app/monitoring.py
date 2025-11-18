# api/app/monitoring.py
"""
Módulo de monitoreo de drift usando Evidently AI 0.7.x.
Detecta drift en datos y performance del modelo.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from evidently import Regression, Report, Dataset, DataDefinition
from evidently.metrics import (
    DriftedColumnsCount,
    ValueDrift,
    MAE,
    RMSE,
    R2Score,
    MAPE,
)


class DriftMonitor:
    """
    Monitorea drift en datos de entrada y predicciones del modelo.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        target_column: str = "usage_kwh",
        prediction_column: str = "prediction",
        numerical_features: Optional[list] = None,
        categorical_features: Optional[list] = None,
    ):
        """
        Args:
            reference_data: Dataset de referencia (datos de entrenamiento)
            target_column: Nombre de la columna target
            prediction_column: Nombre de la columna de predicciones
            numerical_features: Lista de features numéricas
            categorical_features: Lista de features categóricas
        """
        self.reference_data = reference_data
        self.target_column = target_column
        self.prediction_column = prediction_column

        # Auto-detectar features si no se proporcionan
        if numerical_features is None:
            numerical_features = reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            # Remover target y prediction si están presentes
            numerical_features = [
                f for f in numerical_features
                if f not in [target_column, prediction_column]
            ]

        if categorical_features is None:
            categorical_features = reference_data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def generate_data_drift_report(
        self, current_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera reporte de drift en los datos de entrada.

        Args:
            current_data: Datos actuales (producción)
            save_path: Path para guardar reporte HTML (opcional)

        Returns:
            Dict con métricas de drift
        """
        # Crear métricas de drift
        metrics = [DriftedColumnsCount()]

        # Agregar ValueDrift para cada feature numérica
        for col in self.numerical_features[:10]:  # Limitar a 10 para no saturar
            if col in current_data.columns:
                metrics.append(ValueDrift(column=col))

        # Crear y ejecutar reporte
        data_drift_report = Report(metrics=metrics)

        snapshot = data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
        )

        # Guardar HTML si se especifica path
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            snapshot.save_html(save_path)
            print(f"[INFO] Data drift report saved to: {save_path}")

        # Extraer métricas
        report_dict = snapshot.dict()

        # Buscar métricas en la estructura
        drift_detected = False
        num_drifted = 0
        total_cols = len(self.numerical_features)
        drift_by_columns = {}

        for metric in report_dict.get("metrics", []):
            if metric.get("metric") == "DriftedColumnsCount":
                result = metric.get("result", {})
                num_drifted = result.get("number_of_drifted_columns", 0)
                total_cols = result.get("number_of_columns", total_cols)
                drift_detected = num_drifted > 0
            elif metric.get("metric") == "ValueDrift":
                result = metric.get("result", {})
                col_name = result.get("column_name", "unknown")
                drift_by_columns[col_name] = {
                    "drift_detected": result.get("drift_detected", False),
                    "drift_score": result.get("drift_score", 0.0),
                }

        share_drifted = num_drifted / total_cols if total_cols > 0 else 0.0

        drift_summary = {
            "dataset_drift_detected": drift_detected,
            "number_of_drifted_columns": num_drifted,
            "share_of_drifted_columns": share_drifted,
            "drift_by_columns": drift_by_columns,
            "timestamp": datetime.now().isoformat(),
        }

        return drift_summary

    def generate_model_performance_report(
        self, current_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera reporte de performance del modelo (requiere target real).

        Args:
            current_data: Datos con predicciones y targets reales
            save_path: Path para guardar reporte HTML (opcional)

        Returns:
            Dict con métricas de performance
        """
        # Verificar que existan target y predictions
        if self.target_column not in current_data.columns:
            raise ValueError(
                f"Column '{self.target_column}' not found in current_data"
            )
        if self.prediction_column not in current_data.columns:
            raise ValueError(
                f"Column '{self.prediction_column}' not found in current_data"
            )

        # Crear DataDefinition para regresión
        data_definition = DataDefinition(
            regression=[
                Regression(
                    target=self.target_column,
                    prediction=self.prediction_column
                )
            ]
        )

        # Crear Datasets con la definición
        reference_dataset = Dataset.from_pandas(self.reference_data, data_definition)
        current_dataset = Dataset.from_pandas(current_data, data_definition)

        # Crear y ejecutar reporte de regresión
        regression_report = Report(metrics=[
            MAE(),
            RMSE(),
            R2Score(),
            MAPE(),
        ])

        snapshot = regression_report.run(
            reference_data=reference_dataset,
            current_data=current_dataset,
        )

        # Guardar HTML
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            snapshot.save_html(save_path)
            print(f"[INFO] Model performance report saved to: {save_path}")

        # Extraer métricas
        report_dict = snapshot.dict()

        # Buscar métricas de regresión en el reporte
        metrics_summary = {"timestamp": datetime.now().isoformat()}

        for metric in report_dict.get("metrics", []):
            metric_name = metric.get("metric", "")
            result = metric.get("result", {})

            if metric_name == "MAE":
                metrics_summary["mae"] = result.get("current", {}).get("value")
            elif metric_name == "RMSE":
                metrics_summary["rmse"] = result.get("current", {}).get("value")
            elif metric_name == "R2Score":
                metrics_summary["r2_score"] = result.get("current", {}).get("value")
            elif metric_name == "MAPE":
                metrics_summary["mape"] = result.get("current", {}).get("value")

        return metrics_summary

    def generate_comprehensive_report(
        self, current_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera reporte completo con drift + performance.

        Args:
            current_data: Datos actuales con predicciones (y opcionalmente targets)
            save_path: Path para guardar reporte HTML

        Returns:
            Dict con todas las métricas
        """
        # Métricas base de drift
        metrics = [DriftedColumnsCount()]

        # Agregar drift por columnas clave (soporta diferentes formatos)
        key_columns = [
            "lagging_current_reactive.power_kvarh",
            "nsm",
            "co2(tco2)",
            # Compatibilidad con nombres antiguos
            "Lagging_Current_Reactive.Power_kVarh",
            "NSM",
            "CO2(tCO2)"
        ]
        for col in key_columns:
            if col in current_data.columns:
                metrics.append(ValueDrift(column=col))

        # Si hay target y predicción, agregar métricas de regresión
        has_regression = (
            self.target_column in current_data.columns
            and self.prediction_column in current_data.columns
        )

        if has_regression:
            metrics.extend([
                MAE(),
                RMSE(),
                R2Score(),
                MAPE(),
            ])

        # Crear y ejecutar reporte
        comprehensive_report = Report(metrics=metrics)

        # Si hay regresión, usar Dataset con DataDefinition
        if has_regression:
            data_definition = DataDefinition(
                regression=[
                    Regression(
                        target=self.target_column,
                        prediction=self.prediction_column
                    )
                ]
            )
            reference_dataset = Dataset.from_pandas(self.reference_data, data_definition)
            current_dataset = Dataset.from_pandas(current_data, data_definition)

            snapshot = comprehensive_report.run(
                reference_data=reference_dataset,
                current_data=current_dataset,
            )
        else:
            # Sin regresión, usar DataFrames directamente
            snapshot = comprehensive_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
            )

        # Guardar HTML
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            snapshot.save_html(save_path)
            print(f"[INFO] Comprehensive report saved to: {save_path}")

        # Guardar JSON
        if save_path:
            json_path = save_path.replace(".html", ".json")
            snapshot.save_json(json_path)

        # Retornar diccionario
        return snapshot.dict()


def create_monitor_from_csv(
    reference_csv_path: str,
    target_column: str = "usage_kwh",
    prediction_column: str = "prediction",
) -> DriftMonitor:
    """
    Helper para crear monitor desde CSV de entrenamiento.

    Args:
        reference_csv_path: Path al CSV de datos de entrenamiento
        target_column: Nombre de columna target
        prediction_column: Nombre de columna predicción

    Returns:
        DriftMonitor configurado
    """
    reference_data = pd.read_csv(reference_csv_path)

    # Identificar features numéricas (excluir target, date, etc)
    exclude_cols = [target_column, prediction_column, "date"]
    numerical_features = [
        col
        for col in reference_data.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]

    categorical_features = [
        col
        for col in reference_data.select_dtypes(include=["object", "category"]).columns
        if col not in exclude_cols
    ]

    return DriftMonitor(
        reference_data=reference_data,
        target_column=target_column,
        prediction_column=prediction_column,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )
