# steel_energy_refactored_v1.py
# -*- coding: utf-8 -*-

from typing import Iterable, Tuple, Optional, List, Dict
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Constantes
# =========================
TARGET_COL = "usage_kwh"


# =========================
# Utilidades
# =========================
def _resolve_column_name(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """
    Devuelve el nombre de columna existente en df que coincida (ignorando mayúsculas,
    guiones bajos, espacios y puntos) con alguno de los 'candidates'.
    """
    def norm(s: str) -> str:
        return str(s).lower().replace(" ", "").replace("_", "").replace(".", "")
    normalized_cols = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in normalized_cols:
            return normalized_cols[key]
    raise KeyError(f"Ninguna coincidencia para {list(candidates)} en columnas de df.")


def describe_with_percentiles(
    df: pd.DataFrame,
    column: str,
    percentiles: Iterable[float] = (0.05, 0.25, 0.5, 0.75, 0.99),
) -> pd.Series:
    col = _resolve_column_name(df, [column])
    desc = df[col].describe(percentiles=list(percentiles))
    print(desc)
    return desc


def filter_by_threshold(
    df: pd.DataFrame,
    column: str,
    upper: Optional[float] = None,
    lower: Optional[float] = None,
    inclusive: bool = True,
) -> pd.DataFrame:
    col = _resolve_column_name(df, [column])
    mask = pd.Series(True, index=df.index)
    if lower is not None:
        mask &= df[col] >= lower if inclusive else df[col] > lower
    if upper is not None:
        mask &= df[col] <= upper if inclusive else df[col] < upper
    return df[mask].copy()


# =========================
# Limpieza y corrección
# =========================
def clean_and_correct_formats(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    date_col: str = "date",
    index_to_check: int = 32905,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Limpia columnas, corrige tipos (fecha, numéricas y texto), identifica NaNs,
    y muestra verificaciones (sample(20), info()) más la fila index==index_to_check.
    """
    if numeric_cols is None:
        numeric_cols = [
            "usage_kwh",
            "lagging_current_reactive.power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2(tco2)",
            "lagging_current_power_factor",
            "leading_current_power_factor",
            "nsm",
        ]
    if text_cols is None:
        text_cols = ["weekstatus", "day_of_week", "load_type"]

    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    df = df.drop_duplicates().reset_index(drop=True)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower()
                .replace({"nan": np.nan, "none": np.nan, "na": np.nan, "": np.nan})
            )
            df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].str.capitalize()

    print("\nConteo de valores NaN por columna:")
    print(df.isna().sum())

    print("\nMuestra aleatoria de 20 filas (df.sample(20)):")
    try:
        display(df.sample(20))  # si es notebook
    except Exception:
        print(df.sample(20))

    print("\nInformación del DataFrame (df.info()):")
    df.info()

    print(f"\n# Fila con índice == {index_to_check}")
    row_check = df[df.index == index_to_check]
    try:
        display(row_check)
    except Exception:
        print(row_check)

    return df, row_check


# =========================
# Visualizaciones / EDA
# =========================
def plot_box_with_percentile(
    df: pd.DataFrame,
    column: str,
    percentile: float = 0.99,
    title_prefix: str = "Distribución",
    log_y: bool = False,
) -> float:
    col = _resolve_column_name(df, [column])
    p_val = float(df[col].quantile(percentile))
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col].dropna(), vert=True, patch_artist=True)
    if log_y:
        plt.yscale("log")
    plt.axhline(y=p_val, color="red", linestyle="--",
                label=f"Percentil {int(percentile*100)} ({p_val:.3g})")
    plt.title(f"{title_prefix} de {col}")
    plt.ylabel(col + (" (escala log)" if log_y else ""))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()
    return p_val


def analyze_univariate_box(
    df: pd.DataFrame,
    column: str,
    cap_value: Optional[float] = None,
    percentile: float = 0.99,
    log_y: bool = False,
) -> Tuple[pd.DataFrame, float]:
    col = _resolve_column_name(df, [column])

    print(f"\n--- Describe de {col} (antes de filtrar) ---")
    describe_with_percentiles(df, col)

    p_val = plot_box_with_percentile(df, col, percentile=percentile, log_y=log_y)
    print(f"Percentil {int(percentile*100)} de {col}: {p_val:.6g}")

    cut = p_val if cap_value is None else float(cap_value)
    df_filtered = filter_by_threshold(df, col, upper=cut, inclusive=True)

    print(f"\n--- Describe de {col} (después de filtrar <= {cut:.6g}) ---")
    describe_with_percentiles(df_filtered, col)

    plot_box_with_percentile(
        df_filtered, col, percentile=percentile,
        title_prefix="Distribución (filtrado)", log_y=log_y
    )
    print("\nInformación del DataFrame filtrado:")
    df_filtered.info()

    return df_filtered, p_val


def analyze_usage_kwh(df: pd.DataFrame, cap_value: Optional[float] = 157.0) -> pd.DataFrame:
    df_filtrado, p99 = analyze_univariate_box(
        df=df, column=TARGET_COL, cap_value=cap_value, percentile=0.99, log_y=False
    )
    print(f"\nValor p99 observado en {TARGET_COL}: {p99:.6g}")
    return df_filtrado


def analyze_lagging_reactive_kvarh(df: pd.DataFrame) -> None:
    col = _resolve_column_name(
        df, ["Lagging_Current_Reactive.Power_kVarh", "lagging_current_reactive.power_kvarh"]
    )
    print(f"\n--- Describe para {col} ---")
    describe_with_percentiles(df, col)
    plot_box_with_percentile(df, col, percentile=0.99)


def analyze_co2_tco2(df: pd.DataFrame, hard_cap: float = 0.070) -> pd.DataFrame:
    col = _resolve_column_name(df, ["co2(tco2)"])
    print(f"\n--- Describe para {col} ---")
    describe_with_percentiles(df, col, percentiles=(0.05, 0.25, 0.5, 0.75, 0.96, 0.99, 0.995))
    p99 = plot_box_with_percentile(df, col, percentile=0.99)
    print(f"Percentil 99 de {col}: {p99:.6g}")

    df_cap = filter_by_threshold(df, col, upper=hard_cap, inclusive=True)
    print(f"\nFiltrado: {col} <= {hard_cap}")
    df_cap.info()
    plot_box_with_percentile(df_cap, col, percentile=0.99, title_prefix="Distribución (filtrado)")
    return df_cap


def plot_correlation_matrix(
    df: pd.DataFrame,
    include_dtypes: Iterable[str] = ("float64", "int64"),
    title: str = "Matriz de Correlación entre Variables Numéricas",
) -> None:
    num_df = df.select_dtypes(include=list(include_dtypes))
    plt.figure(figsize=(12, 8))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.show()


def plot_box_p99_for(
    df: pd.DataFrame, column: str, percentile: float = 0.99, title_suffix: str = ""
) -> float:
    return plot_box_with_percentile(
        df=df, column=column, percentile=percentile,
        title_prefix=f"Distribución{(' ' + title_suffix) if title_suffix else ''}",
        log_y=False,
    )


def corr_matrix_for(df: pd.DataFrame, title_suffix: str = "") -> None:
    plot_correlation_matrix(
        df=df,
        title=f"Matriz de Correlación entre Variables Numéricas{(' - ' + title_suffix) if title_suffix else ''}",
    )


# =========================
# MAIN (acepta paths CSV)
# =========================
def main(dataset_path_modificado: str, dataset_path_objetivo: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    print(f"\n📂 Cargando dataset MODIFICADO desde: {dataset_path_modificado}")
    df_mod = pd.read_csv(dataset_path_modificado)
    print(f"✅ Dataset modificado cargado con {df_mod.shape[0]} filas y {df_mod.shape[1]} columnas")

    df_obj = None
    if dataset_path_objetivo:
        print(f"\n📂 Cargando dataset OBJETIVO desde: {dataset_path_objetivo}")
        df_obj = pd.read_csv(dataset_path_objetivo)
        print(f"✅ Dataset objetivo cargado con {df_obj.shape[0]} filas y {df_obj.shape[1]} columnas")

    # Limpieza y EDA dataset modificado
    df_clean, row_32905 = clean_and_correct_formats(
        df_mod,
        numeric_cols=[
            "usage_kwh",
            "lagging_current_reactive.power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2(tco2)",
            "lagging_current_power_factor",
            "leading_current_power_factor",
            "nsm",
        ],
        text_cols=["weekstatus", "day_of_week", "load_type"],
        date_col="date",
        index_to_check=32905,
    )

    df_after_usage_cap = analyze_usage_kwh(df_clean, cap_value=157.0)
    analyze_lagging_reactive_kvarh(df_after_usage_cap)
    df_final = analyze_co2_tco2(df_after_usage_cap, hard_cap=0.070)
    corr_matrix_for(df_final, title_suffix="(Dataset Final Modificado)")
    p99_final_target = plot_box_p99_for(df_final, column=TARGET_COL, title_suffix="(Final Modificado)")

    # Si hay dataset objetivo, análisis paralelo
    if df_obj is not None:
        print("\n📊 ANÁLISIS PARA EL DATASET OBJETIVO (ORIGINAL)")
        df_obj_clean, _ = clean_and_correct_formats(
            df_obj,
            numeric_cols=[
                "usage_kwh",
                "lagging_current_reactive.power_kvarh",
                "leading_current_reactive_power_kvarh",
                "co2(tco2)",
                "lagging_current_power_factor",
                "leading_current_power_factor",
                "nsm",
            ],
            text_cols=["weekstatus", "day_of_week", "load_type"],
            date_col="date",
            index_to_check=32905,
        )
        corr_matrix_for(df_obj_clean, title_suffix="(Dataset Original)")
        p99_obj = plot_box_p99_for(df_obj_clean, column=TARGET_COL, title_suffix="(Dataset Original)")
        print(f"\nComparación p99 -> Original: {p99_obj:.3f} | Modificado: {p99_final_target:.3f}")

    print("\n✅ Pipeline completado correctamente.\n")
    return {
        "df_clean": df_clean,
        "row_32905": row_32905,
        "df_after_usage_cap": df_after_usage_cap,
        "df_final": df_final,
    }


# =========================
# Ejemplo de ejecución:
#2 datasets: python steel_energy_refactored_v1.py --mod ../data/raw/steel_energy_modified.csv --obj ../data/raw/steel_energy_original.csv
#1 dataset: python steel_energy_refactored_v1.py --mod ../data/raw/steel_energy_modified.csv 
#CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de limpieza y EDA para Steel Energy.")
    parser.add_argument("--mod", required=True, help="Ruta al CSV MODIFICADO (obligatorio).")
    parser.add_argument("--obj", required=False, help="Ruta al CSV OBJETIVO (opcional).")
    args = parser.parse_args()

    main(dataset_path_modificado=args.mod, dataset_path_objetivo=args.obj)
