"""EDAclearIA — point d'entrée principal.

Modes disponibles :
  --file  <chemin.csv>   : Analyse exploratoire CSV guidée par l'IA.
  --video <chemin.mp4>   : Détection et mesure de vitesse de véhicules.
"""

import argparse
from pathlib import Path

from src.analyzer import analyze


def _run_eda(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.exists():
        print(f"Erreur : fichier introuvable — {path}")
        raise SystemExit(1)
    analyze(path)


def _run_speed(args: argparse.Namespace) -> None:
    # Import ici pour ne pas rendre ultralytics obligatoire pour le mode EDA
    from src.speed_detector import SpeedDetector  # noqa: PLC0415

    detector = SpeedDetector(
        line1_y=args.line1,
        line2_y=args.line2,
        real_distance_m=args.distance,
        model_path=args.model,
        speed_limit_kmh=args.limit,
        conf=args.conf,
        display=args.display,
    )

    results = detector.process_video(
        input_path=args.video,
        output=args.output,
        save_csv=args.csv,
    )

    if results:
        speeds = [r.speed_kmh for r in results]
        print(f"\n--- Résumé ---")
        print(f"Véhicules mesurés : {len(results)}")
        print(f"Vitesse min : {min(speeds)} km/h")
        print(f"Vitesse max : {max(speeds)} km/h")
        print(f"Vitesse moy : {sum(speeds)/len(speeds):.1f} km/h")
        if args.limit > 0:
            over = [s for s in speeds if s > args.limit]
            print(f"Dépassements (>{args.limit} km/h) : {len(over)}")
    else:
        print("Aucune vitesse mesurée. Vérifiez les valeurs de --line1, --line2 et --distance.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDAclearIA — Analyse exploratoire CSV ou détection de vitesse vidéo",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Mode EDA ---
    eda_group = parser.add_argument_group("Mode EDA (CSV)")
    eda_group.add_argument("--file", help="Chemin vers le fichier CSV à analyser")

    # --- Mode vidéo ---
    vid_group = parser.add_argument_group("Mode détection vitesse (vidéo)")
    vid_group.add_argument("--video", help="Chemin vers la vidéo à analyser")
    vid_group.add_argument(
        "--line1", type=int, default=300,
        help="Ordonnée (pixels) de la ligne virtuelle 1 (défaut: 300)",
    )
    vid_group.add_argument(
        "--line2", type=int, default=450,
        help="Ordonnée (pixels) de la ligne virtuelle 2 (défaut: 450)",
    )
    vid_group.add_argument(
        "--distance", type=float, default=8.0,
        help="Distance réelle au sol entre les deux lignes en mètres (défaut: 8.0)",
    )
    vid_group.add_argument(
        "--model", default="yolov8n.pt",
        help="Modèle YOLOv8 (yolov8n.pt, yolov8s.pt, yolov8m.pt…) (défaut: yolov8n.pt)",
    )
    vid_group.add_argument(
        "--limit", type=float, default=50.0,
        help="Seuil d'alerte vitesse en km/h, 0 pour désactiver (défaut: 50.0)",
    )
    vid_group.add_argument(
        "--conf", type=float, default=0.4,
        help="Seuil de confiance YOLOv8 (défaut: 0.4)",
    )
    vid_group.add_argument(
        "--output", default=None,
        help="Chemin de la vidéo annotée de sortie (optionnel)",
    )
    vid_group.add_argument(
        "--csv", default=None,
        help="Chemin du fichier CSV résultats (optionnel)",
    )
    vid_group.add_argument(
        "--display", action="store_true",
        help="Afficher la fenêtre de prévisualisation (nécessite un écran)",
    )

    args = parser.parse_args()

    if args.file and args.video:
        parser.error("Spécifiez --file OU --video, pas les deux.")

    if args.file:
        _run_eda(args)
    elif args.video:
        _run_speed(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
