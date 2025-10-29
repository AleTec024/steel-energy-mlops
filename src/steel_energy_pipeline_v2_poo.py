# steel_energy_pipeline_v2_poo.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Tuple, List

import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Config
# =========================
DEFAULT_TARGET = "usage_kwh"
DEFAULT_CAP_TARGET = 157.0
DEFAULT_CO2_CAP = 0.070
DEFAULT_INDEX_CHECK = 32905


# =========================
# Utilidades base (POO)
# =========================
class BaseUtils:
    @staticmethod
    def _norm(s: str) -> str:
        return str(s).lower().replace(" ", "").replace("_", "").replace(".", "")

    def _resolve_column_name(self, df: pd.DataFrame, candidates: Iterable[str]) -> str:
        mapping = {self._norm(c): c for c in df.columns}
        for cand in candidates:
            key = self._norm(cand)
            if key in mapping:
                return mapping[key]
        raise KeyError(
            f"No se encontró ninguna de {list(candidates)} en columnas: {list(df.columns)}"
        )


# =========================
# Carga y exploración
# =========================
@dataclass
class DataLoader(BaseUtils):
    path: str

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        print(f"✅ Cargado: {self.path} | shape={df.shape}")
        return df


class DataExplorer(BaseUtils):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def head_t(self, n: int = 5) -> None:
        print(self.df.head(n).T)

    def describe(self) -> None:
        print(self.df.describe(include="all"))

    def info(self) -> None:
        print(self.df.info())


# =========================
# Limpieza y corrección
# =========================
@dataclass
class DataCleaner(BaseUtils):
    df: pd.DataFrame
    numeric_cols: Optional[List[str]] = None
    text_cols: Optional[List[str]] = None
    date_col: str = "date"
    verbose: bool = True

    def __post_init__(self):
        if self.numeric_cols is None:
            self.numeric_cols = [
                "usage_kwh",
                "lagging_current_reactive.power_kvarh",
                "leading_current_reactive_power_kvarh",
                "co2(tco2)",
                "lagging_current_power_factor",
                "leading_current_power_factor",
                "nsm",
            ]
        if self.text_cols is None:
            self.text_cols = ["weekstatus", "day_of_week", "load_type"]

    def run(self, index_to_check: int = DEFAULT_INDEX_CHECK) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.df.copy()

        # Normaliza encabezados
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # Duplicados
        df = df.drop_duplicates().reset_index(drop=True)

        # Fecha
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")

        # Numéricas
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Texto
        for col in self.text_cols:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.lower()
                    .replace({"nan": np.nan, "none": np.nan, "na": np.nan, "": np.nan})
                )
                df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].str.capitalize()

        if self.verbose:
            print("\nConteo de NaN por columna:")
            print(df.isna().sum())

            print("\nMuestra aleatoria (20 filas):")
            print(df.sample(min(20, len(df))))

            print("\nInfo del DataFrame:")
            df.info()

            print(f"\n# Fila con índice == {index_to_check}")
        row_check = df[df.index == index_to_check]
        if self.verbose:
            print(row_check)

        return df, row_check


# =========================
# EDA Univariado
# =========================
@dataclass
class UnivariateAnalyzer(BaseUtils):
    df: pd.DataFrame
    show_plots: bool = True

    def describe_with_percentiles(
        self, column: str, percentiles: Iterable[float] = (0.05, 0.25, 0.5, 0.75, 0.99)
    ) -> pd.Series:
        col = self._resolve_column_name(self.df, [column])
        desc = self.df[col].describe(percentiles=list(percentiles))
        print(desc)
        return desc

    def plot_box_with_percentile(
        self, column: str, percentile: float = 0.99, title_prefix: str = "Distribución", log_y: bool = False
    ) -> float:
        col = self._resolve_column_name(self.df, [column])
        pval = float(self.df[col].quantile(percentile))
        if self.show_plots:
            plt.figure(figsize=(6, 4))
            plt.boxplot(self.df[col].dropna(), vert=True, patch_artist=True)
            if log_y:
                plt.yscale("log")
            plt.axhline(y=pval, color="red", linestyle="--", label=f"p{int(percentile*100)}={pval:.3g}")
            plt.title(f"{title_prefix} de {col}")
            plt.ylabel(col + (" (log)" if log_y else ""))
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.show()
        return pval

    def analyze_and_cap(
        self, column: str, cap_value: Optional[float] = None, percentile: float = 0.99
    ) -> Tuple[pd.DataFrame, float]:
        self.describe_with_percentiles(column)
        pval = self.plot_box_with_percentile(column, percentile)
        cut = pval if cap_value is None else float(cap_value)
        col = self._resolve_column_name(self.df, [column])
        df_f = self.df[self.df[col] <= cut].copy()
        print(f"\nDescribe (<= {cut:.6g})")
        print(df_f[col].describe())
        if self.show_plots:
            _ = UnivariateAnalyzer(df_f, show_plots=True).plot_box_with_percentile(
                col, percentile, title_prefix="Distribución (filtrado)"
            )
        print("\nInfo filtrado:")
        df_f.info()
        return df_f, pval


# =========================
# EDA específico por columna
# =========================
@dataclass
class SpecificAnalyzer(BaseUtils):
    df: pd.DataFrame
    show_plots: bool = True

    def analyze_lagging_reactive_kvarh(self) -> None:
        col = self._resolve_column_name(
            self.df, ["Lagging_Current_Reactive.Power_kVarh", "lagging_current_reactive.power_kvarh"]
        )
        UnivariateAnalyzer(self.df, show_plots=self.show_plots).describe_with_percentiles(col)
        UnivariateAnalyzer(self.df, show_plots=self.show_plots).plot_box_with_percentile(col, 0.99)

    def analyze_co2_tco2(self, hard_cap: float = DEFAULT_CO2_CAP) -> pd.DataFrame:
        col = self._resolve_column_name(self.df, ["co2(tco2)"])
        UnivariateAnalyzer(self.df, show_plots=self.show_plots).describe_with_percentiles(
            col, percentiles=(0.05, 0.25, 0.5, 0.75, 0.96, 0.99, 0.995)
        )
        UnivariateAnalyzer(self.df, show_plots=self.show_plots).plot_box_with_percentile(col, 0.99)
        df_cap = self.df[self.df[col] <= hard_cap].copy()
        print(f"\nFiltrado: {col} <= {hard_cap}")
        df_cap.info()
        UnivariateAnalyzer(df_cap, show_plots=self.show_plots).plot_box_with_percentile(
            col, 0.99, title_prefix="Distribución (filtrado)"
        )
        return df_cap


# =========================
# EDA Multivariado
# =========================
@dataclass
class MultivariateAnalyzer(BaseUtils):
    df: pd.DataFrame
    show_plots: bool = True

    def plot_correlation_matrix(
        self, include_dtypes: Iterable[str] = ("float64", "int64"), title: str = "Matriz de Correlación"
    ) -> None:
        num_df = self.df.select_dtypes(include=list(include_dtypes))
        if self.show_plots:
            plt.figure(figsize=(12, 8))
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title(title)
            plt.show()


# =========================
# Pipeline orquestador
# =========================
@dataclass
class SteelEnergyPipeline(BaseUtils):
    target_col: str = DEFAULT_TARGET
    cap_target: float = DEFAULT_CAP_TARGET
    co2_cap: float = DEFAULT_CO2_CAP
    index_check: int = DEFAULT_INDEX_CHECK
    show_plots: bool = True

    def run_for_paths(
        self, path_modificado: str, path_objetivo: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        # 1) Carga
        df_mod = DataLoader(path_modificado).load()
        df_obj = DataLoader(path_objetivo).load() if path_objetivo else None

        # 2) Limpieza
        df_clean, row_32905 = DataCleaner(df_mod, verbose=True).run(index_to_check=self.index_check)

        # 3) Univariado target + cap
        uni = UnivariateAnalyzer(df_clean, show_plots=self.show_plots)
        df_after_cap, p99_target = uni.analyze_and_cap(self.target_col, cap_value=self.cap_target)

        # 4) Específicos
        spec = SpecificAnalyzer(df_after_cap, show_plots=self.show_plots)
        spec.analyze_lagging_reactive_kvarh()
        df_final = spec.analyze_co2_tco2(hard_cap=self.co2_cap)

        # 5) Correlación final
        MultivariateAnalyzer(df_final, show_plots=self.show_plots).plot_correlation_matrix(
            title="Matriz de Correlación (Dataset Final)"
        )

        # 6) Si hay dataset objetivo → limpieza + correlación + comparación p99
        if df_obj is not None:
            df_obj_clean, _ = DataCleaner(df_obj, verbose=True).run(index_to_check=self.index_check)
            MultivariateAnalyzer(df_obj_clean, show_plots=self.show_plots).plot_correlation_matrix(
                title="Matriz de Correlación (Dataset Original)"
            )
            p99_obj = UnivariateAnalyzer(df_obj_clean, show_plots=self.show_plots).plot_box_with_percentile(
                self.target_col, 0.99, "Distribución (Original)"
            )
            print(f"\nComparación p99 de {self.target_col} → Original: {p99_obj:.3f} | Modificado: {p99_target:.3f}")

        print("\n✅ Pipeline POO finalizado.")
        return {
            "df_clean": df_clean,
            "row_32905": row_32905,
            "df_after_usage_cap": df_after_cap,
            "df_final": df_final,
        }


# =========================
# CLI
# =========================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline POO de limpieza y EDA para Steel Energy."
    )
    p.add_argument("--mod", required=True, help="Ruta al CSV MODIFICADO (obligatorio).")
    p.add_argument("--obj", required=False, help="Ruta al CSV OBJETIVO (opcional).")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Nombre de la columna objetivo (default: usage_kwh).")
    p.add_argument("--cap-target", type=float, default=DEFAULT_CAP_TARGET,
                   help=f"Cap (umbral superior) para el target (default: {DEFAULT_CAP_TARGET}).")
    p.add_argument("--co2-cap", type=float, default=DEFAULT_CO2_CAP,
                   help=f"Cap para co2(tco2) (default: {DEFAULT_CO2_CAP}).")
    p.add_argument("--index-check", type=int, default=DEFAULT_INDEX_CHECK,
                   help=f"Índice a mostrar para verificación (default: {DEFAULT_INDEX_CHECK}).")
    p.add_argument("--no-plots", action="store_true", help="Desactiva gráficos (modo headless).")
    return p


def main():
    args = build_argparser().parse_args()

    # Modo headless (sin gráficos)
    show_plots = not args.no_plots
    if not show_plots:
        matplotlib.use("Agg")

    pipe = SteelEnergyPipeline(
        target_col=args.target,
        cap_target=args.cap_target,
        co2_cap=args.co2_cap,
        index_check=args.index_check,
        show_plots=show_plots,
    )

    results = pipe.run_for_paths(
        path_modificado=args.mod,
        path_objetivo=args.obj,
    )

    # aquí podrías guardar CSVs/artefactos si lo deseas.
    # results["df_final"].to_csv("../data/processed/steel_energy_final.csv", index=False)
    #
    # python steel_energy_pipeline_v2_poo.py --mod ../data/raw/steel_energy_modified.csv --obj ../data/raw/steel_energy_original.csv

    #
    # Opcionales:
    # --target usage_kwh  --cap-target 157  --co2-cap 0.07  --no-plots


if __name__ == "__main__":
    main()
