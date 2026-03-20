"""Analyse exploratoire d'un dataset CSV."""

from pathlib import Path

import pandas as pd


def analyze(path: Path) -> str:
    """Charge un CSV, affiche et retourne un résumé statistique."""
    df = pd.read_csv(path)

    lines = []

    lines.append(f"\n=== EDAclearIA : {path.name} ===\n")
    lines.append(f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes\n")

    lines.append("--- Types de colonnes ---")
    lines.append(df.dtypes.to_string())

    lines.append("\n--- Valeurs manquantes ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("Aucune valeur manquante.")
    else:
        lines.append(missing.to_string())

    lines.append("\n--- Doublons ---")
    n_dup = df.duplicated().sum()
    lines.append(f"{n_dup} ligne(s) dupliquée(s).")

    lines.append("\n--- Statistiques descriptives ---")
    lines.append(df.describe(include="all").to_string())

    summary = "\n".join(lines)
    print(summary)
    return summary
