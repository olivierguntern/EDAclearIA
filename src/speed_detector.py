"""Détection de vitesse de véhicules par croisement de lignes virtuelles.

Méthode : YOLOv8 (détection + tracking) + deux lignes virtuelles horizontales.
Vitesse = distance_réelle / (Δframes / fps) × 3,6  →  km/h

Usage rapide :
    detector = SpeedDetector(
        line1_y=300, line2_y=450,
        real_distance_m=8.0,   # distance au sol entre les deux lignes (mètres)
    )
    results = detector.process_video("ma_video.mp4", output="sortie.mp4", save_csv="vitesses.csv")
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# Importation optionnelle — erreur explicite si absent
try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ultralytics est requis : pip install ultralytics"
    ) from exc

# Classes COCO détectées comme véhicules
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle"}

# Couleurs BGR
COLOR_LINE = (0, 255, 255)   # jaune
COLOR_BOX = (0, 200, 0)      # vert
COLOR_SPEED = (255, 255, 255)  # blanc
COLOR_ALERT = (0, 0, 255)    # rouge (dépassement)


@dataclass
class _TrackState:
    """État interne d'un objet tracké."""
    line1_frame: Optional[int] = None
    line2_frame: Optional[int] = None
    speed_kmh: Optional[float] = None
    last_box: Optional[tuple] = None   # (x1, y1, x2, y2)
    display_until: int = 0             # frame jusqu'à laquelle afficher la vitesse
    vehicle_type: str = ""             # classe YOLO (car, truck, bus…)


@dataclass
class SpeedResult:
    """Résultat de vitesse pour un véhicule détecté."""
    track_id: int
    speed_kmh: float
    frame_detected: int
    timestamp_s: float
    vehicle_type: str = ""    # car, truck, bus, motorbike…
    wall_time: str = ""       # heure réelle HH:MM:SS au moment de la détection


class SpeedDetector:
    """Détecteur de vitesse par croisement de deux lignes virtuelles.

    Args:
        line1_y: Ordonnée (pixels) de la première ligne (entrée).
        line2_y: Ordonnée (pixels) de la deuxième ligne (sortie).
        real_distance_m: Distance réelle au sol entre les deux lignes (mètres).
        model_path: Chemin ou nom du modèle YOLOv8 (défaut : yolov8n.pt).
        speed_limit_kmh: Seuil d'alerte en km/h (affiché en rouge si dépassé). 0 = désactivé.
        conf: Seuil de confiance pour la détection (0–1).
        display: Afficher la fenêtre de prévisualisation en temps réel.
    """

    def __init__(
        self,
        line1_y: int,
        line2_y: int,
        real_distance_m: float,
        model_path: str = "yolov8n.pt",
        speed_limit_kmh: float = 50.0,
        conf: float = 0.4,
        display: bool = False,
    ) -> None:
        if line1_y == line2_y:
            raise ValueError("line1_y et line2_y doivent être différents.")
        if real_distance_m <= 0:
            raise ValueError("real_distance_m doit être > 0.")

        self.line1_y = min(line1_y, line2_y)
        self.line2_y = max(line1_y, line2_y)
        self.real_distance_m = real_distance_m
        self.speed_limit_kmh = speed_limit_kmh
        self.conf = conf
        self.display = display

        self._model = YOLO(model_path)
        self._tracks: dict[int, _TrackState] = {}
        self._results: list[SpeedResult] = []
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # démarre non-pausé

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Contrôle du flux (appelable depuis un autre thread)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Interrompt le traitement en cours."""
        self._stop_event.set()

    def pause(self) -> None:
        """Met en pause le traitement."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Reprend le traitement après une pause."""
        self._pause_event.set()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def process_video(
        self,
        input_path: str | Path,
        output: Optional[str | Path] = None,
        save_csv: Optional[str | Path] = None,
        frame_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
        result_callback: Optional[Callable[["SpeedResult"], None]] = None,
    ) -> list[SpeedResult]:
        """Traite une vidéo et retourne la liste des mesures de vitesse.

        Args:
            input_path: Chemin de la vidéo source.
            output: Chemin de la vidéo annotée à sauvegarder (optionnel).
            save_csv: Chemin du fichier CSV résultats (optionnel).
            frame_callback: Appelé après chaque frame annotée avec
                ``(frame_bgr, frame_idx, total_frames)``. Utilisé par le GUI.
            result_callback: Appelé à chaque nouvelle mesure de vitesse.

        Returns:
            Liste de :class:`SpeedResult` (un par véhicule mesuré).
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Vidéo introuvable : {input_path}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo : {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = self._make_writer(output, fps, width, height) if output else None
        self._tracks.clear()
        self._results.clear()
        self._stop_event.clear()
        self._pause_event.set()

        frame_idx = 0
        t0 = time.time()

        try:
            while True:
                if self._stop_event.is_set():
                    break

                # Blocage si pause active
                self._pause_event.wait()

                ok, frame = cap.read()
                if not ok:
                    break

                annotated = self._process_frame(frame, frame_idx, fps, result_callback)

                if writer:
                    writer.write(annotated)

                if frame_callback:
                    frame_callback(annotated, frame_idx, total_frames)

                if self.display:
                    cv2.imshow("Speed Detection — q pour quitter", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
                if frame_idx % 100 == 0:
                    elapsed = time.time() - t0
                    pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                    print(f"  {frame_idx}/{total_frames} frames ({pct:.0f}%) — {elapsed:.1f}s")

        finally:
            cap.release()
            if writer:
                writer.release()
            if self.display:
                cv2.destroyAllWindows()

        print(f"\nTraitement terminé : {frame_idx} frames, {len(self._results)} vitesses mesurées.")

        if save_csv:
            self._write_csv(save_csv)

        return list(self._results)

    # ------------------------------------------------------------------
    # Traitement par frame
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        result_callback: Optional[Callable[["SpeedResult"], None]] = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Lignes virtuelles
        cv2.line(annotated, (0, self.line1_y), (w, self.line1_y), COLOR_LINE, 2)
        cv2.line(annotated, (0, self.line2_y), (w, self.line2_y), COLOR_LINE, 2)
        cv2.putText(annotated, "L1", (5, self.line1_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LINE, 1)
        cv2.putText(annotated, "L2", (5, self.line2_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LINE, 1)

        # Inférence YOLOv8 avec tracking
        results = self._model.track(frame, persist=True, conf=self.conf, verbose=False)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self._model.model.names[cls_id]
                if cls_name not in VEHICLE_CLASSES:
                    continue

                if box.id is None:
                    continue
                track_id = int(box.id[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cy = (y1 + y2) // 2  # centroïde vertical

                state = self._tracks.setdefault(track_id, _TrackState())
                state.last_box = (x1, y1, x2, y2)
                state.vehicle_type = cls_name

                # Franchissement ligne 1
                if state.line1_frame is None and abs(cy - self.line1_y) < 15:
                    state.line1_frame = frame_idx

                # Franchissement ligne 2
                if (
                    state.line1_frame is not None
                    and state.line2_frame is None
                    and abs(cy - self.line2_y) < 15
                    and frame_idx > state.line1_frame
                ):
                    state.line2_frame = frame_idx
                    delta_frames = state.line2_frame - state.line1_frame
                    speed = (self.real_distance_m / (delta_frames / fps)) * 3.6
                    state.speed_kmh = round(speed, 1)
                    state.display_until = frame_idx + int(fps * 3)  # affiche 3 s

                    result = SpeedResult(
                        track_id=track_id,
                        speed_kmh=state.speed_kmh,
                        frame_detected=frame_idx,
                        timestamp_s=round(frame_idx / fps, 2),
                        vehicle_type=state.vehicle_type,
                        wall_time=datetime.now().strftime("%H:%M:%S"),
                    )
                    self._results.append(result)
                    print(
                        f"  [ID {track_id}] {state.vehicle_type}  "
                        f"{state.speed_kmh} km/h  "
                        f"(t={result.timestamp_s}s  {result.wall_time})"
                    )
                    if result_callback:
                        result_callback(result)

                # Dessin boîte englobante
                color = (
                    COLOR_ALERT
                    if (state.speed_kmh and self.speed_limit_kmh > 0 and state.speed_kmh > self.speed_limit_kmh)
                    else COLOR_BOX
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated, f"ID{track_id} {cls_name}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

                # Affichage vitesse
                if state.speed_kmh is not None and frame_idx <= state.display_until:
                    label = f"{state.speed_kmh} km/h"
                    if self.speed_limit_kmh > 0 and state.speed_kmh > self.speed_limit_kmh:
                        label += " !"
                    cv2.putText(
                        annotated, label,
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )

        # Compteur en haut à gauche
        cv2.putText(
            annotated,
            f"Mesures: {len(self._results)}  Frame: {frame_idx}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SPEED, 2,
        )

        return annotated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_writer(path: str | Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Impossible de créer la vidéo de sortie : {path}")
        return writer

    def _write_csv(self, path: str | Path) -> None:
        path = Path(path)
        fieldnames = ["track_id", "type_vehicule", "heure_passage", "speed_kmh", "frame", "timestamp_s"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._results:
                writer.writerow({
                    "track_id": r.track_id,
                    "type_vehicule": r.vehicle_type,
                    "heure_passage": r.wall_time,
                    "speed_kmh": r.speed_kmh,
                    "frame": r.frame_detected,
                    "timestamp_s": r.timestamp_s,
                })
        print(f"Résultats sauvegardés → {path}")

    # ------------------------------------------------------------------
    # Utilitaire : aide à la calibration
    # ------------------------------------------------------------------

    @staticmethod
    def calibrate_from_frame(frame_path: str | Path) -> None:
        """Ouvre une frame et affiche les coordonnées au clic (aide calibration).

        Utilisation : cliquer sur les 2 repères au sol → noter les Y pixels.
        Appuyer sur 'q' pour quitter.
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Image introuvable : {frame_path}")

        clicks: list[tuple[int, int]] = []

        def on_click(event, x, y, flags, param):  # noqa: ANN001
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                print(f"  Clic {len(clicks)} : x={x}, y={y}")
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                cv2.imshow("Calibration", frame)

        cv2.imshow("Calibration", frame)
        cv2.setMouseCallback("Calibration", on_click)
        print("Cliquez sur vos 2 repères au sol, puis appuyez sur 'q'.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(clicks) >= 2:
            y_values = sorted(c[1] for c in clicks)
            print(f"\nSuggestion : line1_y={y_values[0]}, line2_y={y_values[-1]}")
