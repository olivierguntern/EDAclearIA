"""EDAclearIA — point d'entrée principal."""

import argparse
from pathlib import Path

from src.analyzer import analyze
from src.ai_interpreter import interpret


def main():
    parser = argparse.ArgumentParser(description="EDAclearIA — Analyse exploratoire guidée par l'IA")
    parser.add_argument("--file", required=True, help="Chemin vers le fichier CSV à analyser")
    parser.add_argument("--no-ai", action="store_true", help="Désactiver l'interprétation IA")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Erreur : fichier introuvable — {path}")
        raise SystemExit(1)

    summary = analyze(path)

    if not args.no_ai:
        print("\n=== Interprétation IA (Claude) ===\n")
        interpretation = interpret(summary)
        print(interpretation)


if __name__ == "__main__":
    main()
