# api/app/data_simulator.py
"""
Generador de datos sintéticos con drift simulado para testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Literal


class DriftSimulator:
    """
    Simula drift en datos de producción para testing de monitoreo.
    """

    def __init__(self, reference_data: pd.DataFrame, random_seed: int = 42):
        """
        Args:
            reference_data: Dataset de referencia original
            random_seed: Semilla para reproducibilidad
        """
        self.reference_data = reference_data.copy()
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def simulate_covariate_drift(
        self,
        n_samples: int = 100,
        drift_magnitude: float = 0.3,
        columns_to_drift: list = None,
    ) -> pd.DataFrame:
        """
        Simula drift de covariables (cambios en distribución de features).

        Args:
            n_samples: Número de muestras a generar
            drift_magnitude: Magnitud del drift (0.0 = sin drift, 1.0 = drift extremo)
            columns_to_drift: Columnas específicas para aplicar drift

        Returns:
            DataFrame con drift simulado
        """
        # Muestrear desde datos de referencia
        sampled = self.reference_data.sample(n=n_samples, replace=True, random_state=self.random_seed)
        drifted_data = sampled.copy()

        # Identificar columnas numéricas
        if columns_to_drift is None:
            columns_to_drift = drifted_data.select_dtypes(include=[np.number]).columns.tolist()
            # Excluir target si existe
            columns_to_drift = [c for c in columns_to_drift if c not in ["Usage_kWh", "usage_kwh"]]

        print(f"[INFO] Applying covariate drift to columns: {columns_to_drift}")

        for col in columns_to_drift:
            if col in drifted_data.columns and pd.api.types.is_numeric_dtype(drifted_data[col]):
                # Calcular estadísticas originales
                mean = drifted_data[col].mean()
                std = drifted_data[col].std()

                # Agregar sesgo (shift en media)
                shift = drift_magnitude * std * np.random.choice([-1, 1])
                drifted_data[col] = drifted_data[col] + shift

                # Cambiar varianza
                scale_factor = 1 + (drift_magnitude * 0.5 * np.random.uniform(-1, 1))
                drifted_data[col] = mean + (drifted_data[col] - mean) * scale_factor

                print(f"  - {col}: shift={shift:.2f}, scale={scale_factor:.2f}")

        return drifted_data

    def simulate_concept_drift(
        self, n_samples: int = 100, noise_factor: float = 0.3
    ) -> pd.DataFrame:
        """
        Simula concept drift (cambio en relación feature-target).
        Agrega ruido a las predicciones para simular degradación del modelo.

        Args:
            n_samples: Número de muestras
            noise_factor: Factor de ruido (0.0 = sin ruido, 1.0 = mucho ruido)

        Returns:
            DataFrame con concept drift
        """
        sampled = self.reference_data.sample(n=n_samples, replace=True, random_state=self.random_seed)
        drifted_data = sampled.copy()

        # Si existe target, agregar ruido (soporta ambos nombres)
        target_col = None
        if "usage_kwh" in drifted_data.columns:
            target_col = "usage_kwh"
        elif "Usage_kWh" in drifted_data.columns:
            target_col = "Usage_kWh"

        if target_col:
            target_std = drifted_data[target_col].std()
            noise = np.random.normal(0, noise_factor * target_std, size=n_samples)
            drifted_data[target_col] = drifted_data[target_col] + noise

            print(f"[INFO] Concept drift applied: noise_std={noise_factor * target_std:.2f}")

        return drifted_data

    def simulate_sudden_drift(
        self,
        n_samples: int = 100,
        drift_type: Literal["high_load", "low_load", "anomaly"] = "high_load",
    ) -> pd.DataFrame:
        """
        Simula drift abrupto (cambio repentino en régimen de operación).

        Args:
            n_samples: Número de muestras
            drift_type: Tipo de drift abrupto

        Returns:
            DataFrame con drift abrupto
        """
        sampled = self.reference_data.sample(n=n_samples, replace=True, random_state=self.random_seed)
        drifted_data = sampled.copy()

        if drift_type == "high_load":
            # Simular período de alta carga (aumentar consumo y potencia reactiva)
            numeric_cols = drifted_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if "power" in col.lower() or "usage" in col.lower():
                    drifted_data[col] = drifted_data[col] * np.random.uniform(1.5, 2.0)

            print("[INFO] Sudden drift: HIGH_LOAD scenario")

        elif drift_type == "low_load":
            # Simular período de baja carga
            numeric_cols = drifted_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if "power" in col.lower() or "usage" in col.lower():
                    drifted_data[col] = drifted_data[col] * np.random.uniform(0.3, 0.6)

            print("[INFO] Sudden drift: LOW_LOAD scenario")

        elif drift_type == "anomaly":
            # Inyectar valores anómalos en algunas muestras
            n_anomalies = int(n_samples * 0.1)  # 10% de anomalías
            anomaly_indices = np.random.choice(drifted_data.index, n_anomalies, replace=False)

            numeric_cols = drifted_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_mean = drifted_data[col].mean()
                col_std = drifted_data[col].std()
                # Valores extremos (±3 a ±5 desviaciones estándar)
                anomaly_values = col_mean + np.random.choice([-1, 1], n_anomalies) * np.random.uniform(3, 5, n_anomalies) * col_std
                drifted_data.loc[anomaly_indices, col] = anomaly_values

            print(f"[INFO] Sudden drift: ANOMALY scenario ({n_anomalies} anomalies injected)")

        return drifted_data

    def simulate_gradual_drift(
        self, n_samples: int = 500, drift_speed: float = 0.5
    ) -> pd.DataFrame:
        """
        Simula drift gradual (cambio progresivo en el tiempo).

        Args:
            n_samples: Número de muestras
            drift_speed: Velocidad del drift (0.0 = lento, 1.0 = rápido)

        Returns:
            DataFrame con drift gradual
        """
        sampled = self.reference_data.sample(n=n_samples, replace=True, random_state=self.random_seed)
        drifted_data = sampled.copy()

        # Identificar columnas numéricas
        numeric_cols = [c for c in drifted_data.select_dtypes(include=[np.number]).columns if c != "Usage_kWh"]

        # Aplicar drift progresivo
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(drifted_data[col]):
                mean = drifted_data[col].mean()
                std = drifted_data[col].std()

                # Crear gradiente temporal
                drift_gradient = np.linspace(0, drift_speed * std, n_samples)
                drifted_data[col] = drifted_data[col] + drift_gradient

        print(f"[INFO] Gradual drift applied over {n_samples} samples")

        return drifted_data

    def create_drift_scenario(
        self,
        scenario: Literal["no_drift", "mild_drift", "moderate_drift", "severe_drift"],
        n_samples: int = 200,
    ) -> pd.DataFrame:
        """
        Crea un escenario de drift predefinido.

        Args:
            scenario: Nivel de drift deseado
            n_samples: Número de muestras

        Returns:
            DataFrame según el escenario
        """
        scenarios = {
            "no_drift": lambda: self.reference_data.sample(n=n_samples, replace=True),
            "mild_drift": lambda: self.simulate_covariate_drift(n_samples, drift_magnitude=0.2),
            "moderate_drift": lambda: self.simulate_covariate_drift(n_samples, drift_magnitude=0.5),
            "severe_drift": lambda: self.simulate_covariate_drift(n_samples, drift_magnitude=0.8),
        }

        if scenario not in scenarios:
            raise ValueError(f"Scenario '{scenario}' not recognized. Choose from: {list(scenarios.keys())}")

        drifted_data = scenarios[scenario]()
        print(f"[INFO] Created scenario: {scenario.upper()} with {n_samples} samples")

        return drifted_data


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos de referencia
    reference_df = pd.read_csv("../../data/raw/steel_energy_modified.csv")

    # Crear simulador
    simulator = DriftSimulator(reference_df)

    # Generar diferentes tipos de drift
    print("\n=== Generating drift scenarios ===")

    no_drift = simulator.create_drift_scenario("no_drift", n_samples=100)
    mild_drift = simulator.create_drift_scenario("mild_drift", n_samples=100)
    severe_drift = simulator.create_drift_scenario("severe_drift", n_samples=100)

    print("\n✓ Drift scenarios created successfully")
