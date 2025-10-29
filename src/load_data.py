# src/load_data.py
# -*- coding: utf-8 -*-
"""
Módulo para cargar y explorar datasets CSV.
"""

import pandas as pd
import argparse


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga un dataset CSV en un DataFrame de pandas.
    
    Parámetros
    ----------
    filepath : str
        Ruta al archivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame cargado.
    """
    data = pd.read_csv(filepath)
    print(f"✅ Dataset cargado correctamente desde: {filepath}")
    print(f"📊 Filas: {data.shape[0]}, Columnas: {data.shape[1]}")
    return data


def explore_data(data: pd.DataFrame) -> None:
    """
    Muestra una exploración inicial del dataset.
    
    Incluye:
    - Primeras filas transpuestas (head().T)
    - Estadísticas descriptivas (describe())
    - Información general (info())
    """
    print("\n🔍 Primeras filas (transpuestas):")
    print(data.head().T)

    print("\n📈 Estadísticas descriptivas:")
    print(data.describe(include="all"))

    print("\nℹ️ Información del DataFrame:")
    print(data.info())


def load_and_explore(filepath: str) -> pd.DataFrame:
    """
    Carga un CSV y muestra su exploración básica.
    """
    data = load_data(filepath)
    explore_data(data)
    return data


# ========== MAIN ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carga y exploración inicial de dataset CSV.")
    parser.add_argument("--csv", required=True, help="Ruta al archivo CSV a cargar.")
    args = parser.parse_args()

    df = load_and_explore(args.csv)

#usarlo:
#Desde otro módulo o notebook:
#from src.load_data import load_data, explore_data
#df = load_data("../../data/raw/steel_energy_modified.csv")
#explore_data(df)
#Desde terminal:
#python src/load_data.py --csv ../../data/raw/steel_energy_modified.csv
