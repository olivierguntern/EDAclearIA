"""EDAclearIA — point d'entrée principal."""

import argparse
from pathlib import Path

from src.analyzer import analyze


def main():
    parser = argparse.ArgumentParser(description="EDAclearIA — Analyse exploratoire guidée par l'IA")
    parser.add_argument("--file", required=True, help="Chemin vers le fichier CSV à analyser")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Erreur : fichier introuvable — {path}")
        raise SystemExit(1)

    analyze(path)


if __name__ == "__main__":
    main()
