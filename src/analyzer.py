"""Analyse exploratoire d'un dataset CSV."""

from pathlib import Path

import pandas as pd


def analyze(path: Path) -> None:
    """Charge un CSV et affiche un résumé statistique."""
    df = pd.read_csv(path)

    print(f"\n=== EDAclearIA : {path.name} ===\n")
    print(f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes\n")

    print("--- Types de colonnes ---")
    print(df.dtypes.to_string())

    print("\n--- Valeurs manquantes ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Aucune valeur manquante.")
    else:
        print(missing.to_string())

    print("\n--- Statistiques descriptives ---")
    print(df.describe(include="all").to_string())
