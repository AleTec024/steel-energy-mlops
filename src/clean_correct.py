# clean_correct.py
import argparse
import numpy as np
import pandas as pd

def clean_and_correct_formats(
    df: pd.DataFrame,
    numeric_cols=None,
    text_cols=None,
    date_col: str = "date",
    index_to_check: int = 32905,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if numeric_cols is None:
        numeric_cols = [
            "usage_kwh",
            "lagging_current_reactive.power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2(tco2)", "lagging_current_power_factor",
            "leading_current_power_factor", "nsm",
        ]
    if text_cols is None:
        text_cols = ["weekstatus", "day_of_week", "load_type"]

    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_"))
    df = df.drop_duplicates().reset_index(drop=True)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in text_cols:
        if c in df.columns:
            df[c] = (df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                            .str.lower().replace({"nan": np.nan, "none": np.nan, "na": np.nan, "": np.nan}))
            df.loc[df[c].notna(), c] = df.loc[df[c].notna(), c].str.capitalize()

    print("\nConteo de NaN por columna:\n", df.isna().sum())
    print("\nMuestra (20):\n", df.sample(min(20, len(df))))
    print("\nInfo:")
    df.info()

    row_check = df[df.index == index_to_check]
    print(f"\nFila con índice == {index_to_check}:\n", row_check)
    return df, row_check

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al CSV de entrada")
    ap.add_argument("--out", required=False, help="Ruta para guardar CSV limpio")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    clean, _ = clean_and_correct_formats(raw)

    if args.out:
        clean.to_csv(args.out, index=False)
        print(f"💾 Guardado CSV limpio en: {args.out}")


#ejecutarlo:
#python src/clean_correct.py --csv ../../data/raw/steel_energy_modified.csv --out ../../data/processed/steel_clean.csv
