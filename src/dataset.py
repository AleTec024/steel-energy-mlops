import pandas as pd
import os

def load_and_clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # 1. Eliminar duplicados
    df.drop_duplicates(inplace=True)

    # 2. Manejar valores nulos
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 3. Guardar limpio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Datos limpios guardados en: {output_path}")

if __name__ == "__main__":
    load_and_clean_data(
        "data/raw/Steel_industry_data.csv",
        "data/processed/clean_steel_data.csv"
    )
