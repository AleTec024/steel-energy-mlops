#!/usr/bin/env python3
"""
Script de demostración de monitoreo de drift con Evidently AI.

Este script:
1. Carga datos de referencia
2. Simula diferentes escenarios de drift
3. Genera reportes HTML visuales con Evidently
4. Muestra métricas de drift en consola

Uso:
    python demo_drift_monitoring.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar path para imports
sys.path.insert(0, str(Path(__file__).parent / "api"))

from api.app.monitoring import DriftMonitor
from api.app.data_simulator import DriftSimulator


def print_section(title: str):
    """Helper para imprimir secciones bonitas."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_basic_drift_detection():
    """Demuestra detección básica de drift."""
    print_section("1. DETECCIÓN BÁSICA DE DRIFT")

    # Cargar datos de referencia (dataset limpio)
    reference_csv = "data/clean/steel_energy_cleaned_v2.csv"
    print(f"📂 Cargando datos de referencia: {reference_csv}")
    reference_df = pd.read_csv(reference_csv)
    print(f"   → Shape: {reference_df.shape}")

    # Crear simulador
    simulator = DriftSimulator(reference_df, random_seed=42)

    # Generar diferentes escenarios
    scenarios = ["no_drift", "mild_drift", "moderate_drift", "severe_drift"]
    results = []

    for scenario in scenarios:
        print(f"\n🔬 Generando escenario: {scenario.upper()}")
        current_df = simulator.create_drift_scenario(scenario, n_samples=200)

        # Crear monitor
        monitor = DriftMonitor(
            reference_data=reference_df.sample(500, random_state=42),
            target_column="Usage_kWh",
        )

        # Analizar drift
        report_path = f"reports/drift/demo_{scenario}_report.html"
        drift_summary = monitor.generate_data_drift_report(
            current_data=current_df, save_path=report_path
        )

        # Mostrar resultados
        print(f"   ✓ Drift detectado: {drift_summary['dataset_drift_detected']}")
        print(
            f"   ✓ Columnas con drift: {drift_summary['number_of_drifted_columns']} "
            f"({drift_summary['share_of_drifted_columns']*100:.1f}%)"
        )
        print(f"   ✓ Reporte guardado: {report_path}")

        results.append(
            {
                "scenario": scenario,
                "drift_detected": drift_summary["dataset_drift_detected"],
                "drifted_columns": drift_summary["number_of_drifted_columns"],
                "share": drift_summary["share_of_drifted_columns"],
            }
        )

    # Resumen en tabla
    print("\n📊 RESUMEN DE DETECCIÓN:")
    print("-" * 80)
    print(f"{'Escenario':<20} {'Drift':<10} {'Columnas':<15} {'Porcentaje':<15}")
    print("-" * 80)
    for r in results:
        drift_emoji = "🔴" if r["drift_detected"] else "🟢"
        print(
            f"{r['scenario']:<20} {drift_emoji:<10} {r['drifted_columns']:<15} {r['share']*100:.1f}%"
        )
    print("-" * 80)


def demo_covariate_drift():
    """Demuestra detección de covariate drift (cambios en distribución de features)."""
    print_section("2. COVARIATE DRIFT - Cambios en Distribución de Features")

    reference_csv = "data/clean/steel_energy_cleaned_v2.csv"
    reference_df = pd.read_csv(reference_csv)
    simulator = DriftSimulator(reference_df, random_seed=42)

    # Simular drift en features específicas (nombres en minúsculas del dataset limpio)
    columns_to_drift = [
        "lagging_current_reactive.power_kvarh",
        "nsm",
        "co2(tco2)",
    ]

    print(f"🎯 Aplicando drift a columnas: {columns_to_drift}")

    current_df = simulator.simulate_covariate_drift(
        n_samples=300, drift_magnitude=0.6, columns_to_drift=columns_to_drift
    )

    # Crear monitor
    monitor = DriftMonitor(
        reference_data=reference_df.sample(500, random_state=42),
        target_column="Usage_kWh",
    )

    # Generar reporte
    report_path = "reports/drift/demo_covariate_drift.html"
    drift_summary = monitor.generate_data_drift_report(
        current_data=current_df, save_path=report_path
    )

    # Mostrar columnas con drift
    print(f"\n📈 Análisis por columna:")
    drift_by_col = drift_summary.get("drift_by_columns", {})

    for col, info in drift_by_col.items():
        if isinstance(info, dict) and "drift_detected" in info:
            drift_status = "🔴 DRIFT" if info["drift_detected"] else "🟢 OK"
            score = info.get("drift_score", 0.0)
            print(f"   {col:<45} {drift_status:<15} Score: {score:.4f}")

    print(f"\n✓ Reporte HTML guardado: {report_path}")


def demo_concept_drift():
    """Demuestra detección de concept drift (cambios en relación X-y)."""
    print_section("3. CONCEPT DRIFT - Degradación de Performance")

    reference_csv = "data/clean/steel_energy_cleaned_v2.csv"
    reference_df = pd.read_csv(reference_csv)

    # Convertir usage_kwh a numérico (por si acaso)
    reference_df["usage_kwh"] = pd.to_numeric(reference_df["usage_kwh"], errors='coerce')
    reference_df = reference_df.dropna(subset=["usage_kwh"])

    # Simular predicciones en datos de referencia (sin error)
    reference_df["prediction"] = reference_df["usage_kwh"] + np.random.normal(
        0, 0.1, len(reference_df)
    )

    simulator = DriftSimulator(reference_df, random_seed=42)

    # Generar datos con concept drift (más ruido en predicciones)
    print("🔄 Simulando degradación del modelo (concept drift)...")
    current_df = simulator.simulate_concept_drift(n_samples=250, noise_factor=0.8)

    # Agregar predicciones ruidosas
    current_df["prediction"] = current_df["usage_kwh"] + np.random.normal(
        0, 2.0, len(current_df)
    )

    # Crear monitor
    monitor = DriftMonitor(
        reference_data=reference_df.sample(500, random_state=42),
        target_column="usage_kwh",
        prediction_column="prediction",
    )

    # Generar reporte de performance
    report_path = "reports/drift/demo_concept_drift_performance.html"
    try:
        performance_summary = monitor.generate_model_performance_report(
            current_data=current_df, save_path=report_path
        )

        print(f"\n📉 Métricas de Performance:")
        print(f"   MAE:  {performance_summary.get('mae', 'N/A'):.4f}")
        print(f"   RMSE: {performance_summary.get('rmse', 'N/A'):.4f}")
        print(f"   R²:   {performance_summary.get('r2_score', 'N/A'):.4f}")
        print(f"\n✓ Reporte guardado: {report_path}")
    except Exception as e:
        print(f"⚠️  Error generando reporte de performance: {e}")


def demo_sudden_drift():
    """Demuestra detección de drift abrupto."""
    print_section("4. SUDDEN DRIFT - Cambios Abruptos en Régimen")

    reference_csv = "data/clean/steel_energy_cleaned_v2.csv"
    reference_df = pd.read_csv(reference_csv)
    simulator = DriftSimulator(reference_df, random_seed=42)

    drift_types = ["high_load", "low_load", "anomaly"]

    for drift_type in drift_types:
        print(f"\n⚡ Simulando: {drift_type.upper()}")
        current_df = simulator.simulate_sudden_drift(n_samples=150, drift_type=drift_type)

        monitor = DriftMonitor(
            reference_data=reference_df.sample(500, random_state=42),
            target_column="Usage_kWh",
        )

        report_path = f"reports/drift/demo_sudden_{drift_type}.html"
        drift_summary = monitor.generate_data_drift_report(
            current_data=current_df, save_path=report_path
        )

        print(
            f"   ✓ Drift detectado: {drift_summary['dataset_drift_detected']} | "
            f"Columnas: {drift_summary['number_of_drifted_columns']} | "
            f"Reporte: {report_path}"
        )


def demo_comprehensive_report():
    """Genera un reporte completo con todas las métricas."""
    print_section("5. REPORTE COMPREHENSIVO")

    reference_csv = "data/clean/steel_energy_cleaned_v2.csv"
    reference_df = pd.read_csv(reference_csv)

    # Convertir usage_kwh a numérico
    reference_df["usage_kwh"] = pd.to_numeric(reference_df["usage_kwh"], errors='coerce')
    reference_df = reference_df.dropna(subset=["usage_kwh"])

    # Agregar predicciones al reference
    reference_df["prediction"] = reference_df["usage_kwh"] + np.random.normal(
        0, 0.2, len(reference_df)
    )

    simulator = DriftSimulator(reference_df, random_seed=42)

    # Crear escenario mixto: drift + degradación
    print("🔬 Generando escenario completo (drift + degradación)...")
    current_df = simulator.simulate_covariate_drift(n_samples=300, drift_magnitude=0.5)
    current_df["prediction"] = current_df["usage_kwh"] + np.random.normal(
        0, 1.5, len(current_df)
    )

    # Crear monitor
    monitor = DriftMonitor(
        reference_data=reference_df.sample(500, random_state=42),
        target_column="usage_kwh",
        prediction_column="prediction",
    )

    # Generar reporte comprehensivo
    report_path = "reports/drift/demo_comprehensive_report.html"
    print(f"📊 Generando reporte comprehensivo...")

    comprehensive_dict = monitor.generate_comprehensive_report(
        current_data=current_df, save_path=report_path
    )

    print(f"   ✓ Reporte HTML guardado: {report_path}")
    print(f"   ✓ Reporte JSON guardado: {report_path.replace('.html', '.json')}")
    print(
        f"   ✓ Total de métricas calculadas: {len(comprehensive_dict.get('metrics', []))}"
    )


def main():
    """Ejecuta todas las demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DEMO: DRIFT MONITORING CON EVIDENTLY AI" + " " * 19 + "║")
    print("║" + " " * 25 + "Steel Energy MLOps Project" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")

    # Crear directorio de reportes
    Path("reports/drift").mkdir(parents=True, exist_ok=True)

    try:
        # Ejecutar demos
        demo_basic_drift_detection()
        demo_covariate_drift()
        demo_concept_drift()
        demo_sudden_drift()
        demo_comprehensive_report()

        # Resumen final
        print_section("✅ DEMOSTRACIÓN COMPLETADA")
        print("📁 Todos los reportes HTML han sido guardados en: reports/drift/")
        print("\n📖 Para ver los reportes, abre cualquier archivo .html en tu navegador:")
        print("   - demo_no_drift_report.html")
        print("   - demo_moderate_drift_report.html")
        print("   - demo_severe_drift_report.html")
        print("   - demo_covariate_drift.html")
        print("   - demo_comprehensive_report.html")
        print("   - Y más...\n")

        print("🚀 PRÓXIMOS PASOS:")
        print("   1. Abre los reportes HTML para ver las visualizaciones interactivas")
        print("   2. Prueba los endpoints de la API en http://localhost:8000/docs")
        print("   3. Usa /monitor/simulate-drift para generar escenarios personalizados")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error durante la demostración: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
