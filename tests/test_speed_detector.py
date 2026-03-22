"""Tests unitaires pour le module speed_detector.

Ces tests vérifient la logique pure (calculs, validations) sans nécessiter
ultralytics ni OpenCV installés — les dépendances lourdes sont mockées.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mocks pour ultralytics et cv2 (non requis pour les tests logiques)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_heavy_deps(tmp_path):
    """Mock ultralytics.YOLO et cv2 avant toute importation."""
    mock_yolo = MagicMock()
    mock_yolo_instance = MagicMock()
    mock_yolo_instance.model.names = {0: "car", 1: "truck", 2: "person"}
    mock_yolo.return_value = mock_yolo_instance

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.return_value.__enter__ = MagicMock()
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.EVENT_LBUTTONDOWN = 1

    ultralytics_mock = MagicMock()
    ultralytics_mock.YOLO = mock_yolo

    with (
        patch.dict(sys.modules, {"ultralytics": ultralytics_mock, "cv2": mock_cv2}),
    ):
        # Forcer le rechargement si déjà importé
        if "src.speed_detector" in sys.modules:
            del sys.modules["src.speed_detector"]
        yield mock_yolo_instance, mock_cv2


# ---------------------------------------------------------------------------
# Tests de validation des paramètres
# ---------------------------------------------------------------------------

class TestSpeedDetectorInit:
    def test_valid_init(self, mock_heavy_deps):
        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=300, line2_y=450, real_distance_m=8.0)
        assert d.line1_y == 300
        assert d.line2_y == 450
        assert d.real_distance_m == 8.0

    def test_lines_swapped_are_normalized(self, mock_heavy_deps):
        """line1_y > line2_y doit être accepté et normalisé."""
        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=450, line2_y=300, real_distance_m=5.0)
        assert d.line1_y == 300
        assert d.line2_y == 450

    def test_same_lines_raises(self, mock_heavy_deps):
        from src.speed_detector import SpeedDetector
        with pytest.raises(ValueError, match="différents"):
            SpeedDetector(line1_y=300, line2_y=300, real_distance_m=5.0)

    def test_zero_distance_raises(self, mock_heavy_deps):
        from src.speed_detector import SpeedDetector
        with pytest.raises(ValueError, match="> 0"):
            SpeedDetector(line1_y=300, line2_y=450, real_distance_m=0.0)

    def test_negative_distance_raises(self, mock_heavy_deps):
        from src.speed_detector import SpeedDetector
        with pytest.raises(ValueError, match="> 0"):
            SpeedDetector(line1_y=300, line2_y=450, real_distance_m=-1.0)


# ---------------------------------------------------------------------------
# Tests de calcul de vitesse
# ---------------------------------------------------------------------------

class TestSpeedCalculation:
    """Vérifie la formule : v = (distance_m / (Δframes / fps)) × 3.6"""

    @pytest.mark.parametrize("distance_m, delta_frames, fps, expected_kmh", [
        # v = (distance_m / (delta_frames / fps)) * 3.6
        (10.0, 10, 25.0, 90.0),   # 10 / (10/25) * 3.6 = 25 * 3.6 = 90.0
        (8.0,  20, 25.0, 36.0),   # 8  / (20/25) * 3.6 = 10 * 3.6 = 36.0
        (5.0,  15, 30.0, 36.0),   # 5  / (15/30) * 3.6 = 10 * 3.6 = 36.0
    ])
    def test_speed_formula(self, distance_m, delta_frames, fps, expected_kmh):
        computed = (distance_m / (delta_frames / fps)) * 3.6
        assert abs(computed - expected_kmh) < 0.01

    def test_50kmh_scenario(self):
        """Voiture à 50 km/h = 13.89 m/s.
        Distance 10 m → temps = 0.72 s.
        À 25 fps → Δframes = 18 frames.
        """
        fps = 25.0
        distance_m = 10.0
        delta_frames = 18
        speed = (distance_m / (delta_frames / fps)) * 3.6
        # 10 / (18/25) * 3.6 = 10 / 0.72 * 3.6 ≈ 50.0
        assert 49.0 <= speed <= 51.0

    def test_30kmh_scenario(self):
        fps = 25.0
        distance_m = 5.0
        delta_frames = 15
        speed = (distance_m / (delta_frames / fps)) * 3.6
        # 5 / (15/25) * 3.6 = 5 / 0.6 * 3.6 = 8.333 * 3.6 = 30.0
        assert abs(speed - 30.0) < 0.1


# ---------------------------------------------------------------------------
# Tests CSV
# ---------------------------------------------------------------------------

class TestCsvOutput:
    def test_write_csv(self, tmp_path, mock_heavy_deps):
        from src.speed_detector import SpeedDetector, SpeedResult
        d = SpeedDetector(line1_y=300, line2_y=450, real_distance_m=8.0)
        d._results = [
            SpeedResult(track_id=1, speed_kmh=52.3, frame_detected=125, timestamp_s=5.0),
            SpeedResult(track_id=2, speed_kmh=38.7, frame_detected=250, timestamp_s=10.0),
        ]
        csv_path = tmp_path / "vitesses.csv"
        d._write_csv(csv_path)

        assert csv_path.exists()
        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 2
        assert rows[0]["speed_kmh"] == "52.3"
        assert rows[1]["track_id"] == "2"
        assert rows[0]["timestamp_s"] == "5.0"

    def test_csv_headers(self, tmp_path, mock_heavy_deps):
        from src.speed_detector import SpeedDetector, SpeedResult
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=6.0)
        d._results = [SpeedResult(track_id=99, speed_kmh=45.0, frame_detected=50, timestamp_s=2.0)]
        csv_path = tmp_path / "out.csv"
        d._write_csv(csv_path)

        with csv_path.open() as f:
            headers = f.readline().strip().split(",")
        assert headers == ["track_id", "speed_kmh", "frame", "timestamp_s"]


# ---------------------------------------------------------------------------
# Tests process_video — erreurs d'entrée
# ---------------------------------------------------------------------------

class TestProcessVideoErrors:
    def test_missing_video_raises(self, mock_heavy_deps):
        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=300, line2_y=450, real_distance_m=8.0)
        with pytest.raises(FileNotFoundError):
            d.process_video("/tmp/__inexistant_video_xyz__.mp4")

    def test_video_not_openable_raises(self, tmp_path, mock_heavy_deps):
        _, mock_cv2 = mock_heavy_deps
        # Simule VideoCapture.isOpened() → False
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap_mock

        # Crée un fichier vide pour passer le FileNotFoundError
        fake_video = tmp_path / "fake.mp4"
        fake_video.write_bytes(b"")

        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=300, line2_y=450, real_distance_m=8.0)
        with pytest.raises(RuntimeError, match="Impossible d'ouvrir"):
            d.process_video(fake_video)
