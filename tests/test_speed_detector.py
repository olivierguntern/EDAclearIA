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
        assert headers == ["track_id", "type_vehicule", "heure_passage", "speed_kmh", "plaque", "photo_path", "frame", "timestamp_s"]


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


# ---------------------------------------------------------------------------
# Helpers pour créer des mocks de bounding boxes YOLO
# ---------------------------------------------------------------------------

def _mock_box(x1: float, y1: float, x2: float, y2: float,
              track_id: int, cls_id: int = 0) -> MagicMock:
    """Crée un mock de bounding box YOLO avec les attributs nécessaires."""
    box = MagicMock()
    xyxy_item = MagicMock()
    xyxy_item.tolist.return_value = [float(x1), float(y1), float(x2), float(y2)]
    box.xyxy.__getitem__ = MagicMock(return_value=xyxy_item)
    box.id = MagicMock()
    box.id.__getitem__ = MagicMock(return_value=track_id)
    box.cls = MagicMock()
    box.cls.__getitem__ = MagicMock(return_value=cls_id)
    return box


def _setup_yolo_track(mock_yolo_inst: MagicMock, boxes: list) -> None:
    """Configure mock_yolo_inst.track() pour retourner une liste de boxes."""
    result_mock = MagicMock()
    result_mock.boxes = boxes
    mock_yolo_inst.track.return_value = [result_mock]


# ---------------------------------------------------------------------------
# Tests de _process_frame — logique de détection
# ---------------------------------------------------------------------------

class TestProcessFrame:
    """Teste la logique interne de franchissement de lignes et calcul de vitesse."""

    def test_no_detections_returns_annotated(self, mock_heavy_deps):
        """Aucune détection → frame annotée retournée, _results vide."""
        mock_yolo, mock_cv2 = mock_heavy_deps
        result_mock = MagicMock()
        result_mock.boxes = None
        mock_yolo.track.return_value = [result_mock]

        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = d._process_frame(frame, 0, 25.0)

        assert len(d._results) == 0
        assert out.shape == frame.shape

    def test_non_vehicle_class_ignored(self, mock_heavy_deps):
        """Classe 'person' (id=2) ne déclenche aucun enregistrement."""
        mock_yolo, _ = mock_heavy_deps
        # cls_id=2 → "person" (dans le mock names = {0:"car",1:"truck",2:"person"})
        box = _mock_box(100, 190, 200, 215, track_id=1, cls_id=2)
        _setup_yolo_track(mock_yolo, [box])

        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d._process_frame(frame, 5, 25.0)

        assert len(d._tracks) == 0

    def test_no_track_id_skipped(self, mock_heavy_deps):
        """Véhicule sans ID de tracking (box.id is None) → ignoré."""
        mock_yolo, _ = mock_heavy_deps
        box = _mock_box(100, 190, 200, 215, track_id=1, cls_id=0)
        box.id = None  # pas encore tracké
        _setup_yolo_track(mock_yolo, [box])

        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d._process_frame(frame, 5, 25.0)

        assert len(d._tracks) == 0

    def test_line1_crossing_recorded(self, mock_heavy_deps):
        """Centroïde proche de line1_y → line1_frame enregistré."""
        mock_yolo, _ = mock_heavy_deps
        # cy = (190+215)//2 = 202, line1_y=200 → |202-200|=2 < 15
        box = _mock_box(100, 190, 200, 215, track_id=5, cls_id=0)
        _setup_yolo_track(mock_yolo, [box])

        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d._process_frame(frame, 10, 25.0)

        assert 5 in d._tracks
        assert d._tracks[5].line1_frame == 10
        assert d._tracks[5].line2_frame is None

    def test_speed_calculated_on_line2_crossing(self, mock_heavy_deps):
        """Franchissement L1 puis L2 → vitesse calculée, résultat ajouté."""
        mock_yolo, _ = mock_heavy_deps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        from src.speed_detector import SpeedDetector
        # distance=10m, Δframes=18, fps=25 → v = (10/(18/25))*3.6 ≈ 50 km/h
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=10.0)

        # Frame 10 : croise ligne1 (cy=202)
        _setup_yolo_track(mock_yolo, [_mock_box(100, 190, 200, 215, track_id=7)])
        d._process_frame(frame, 10, 25.0)
        assert d._tracks[7].line1_frame == 10

        # Frame 28 : croise ligne2 (cy = (388+415)//2 = 401)
        _setup_yolo_track(mock_yolo, [_mock_box(100, 388, 200, 415, track_id=7)])
        d._process_frame(frame, 28, 25.0)

        assert len(d._results) == 1
        assert d._tracks[7].line2_frame == 28
        assert 49.0 <= d._results[0].speed_kmh <= 51.0
        assert d._results[0].track_id == 7
        assert d._results[0].vehicle_type == "car"

    def test_result_callback_called(self, mock_heavy_deps):
        """result_callback est appelé exactement une fois après le franchissement L2."""
        mock_yolo, _ = mock_heavy_deps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=10.0)
        received = []

        _setup_yolo_track(mock_yolo, [_mock_box(100, 190, 200, 215, track_id=3)])
        d._process_frame(frame, 10, 25.0, result_callback=received.append)
        assert len(received) == 0  # pas encore de vitesse

        _setup_yolo_track(mock_yolo, [_mock_box(100, 388, 200, 415, track_id=3)])
        d._process_frame(frame, 28, 25.0, result_callback=received.append)
        assert len(received) == 1
        assert received[0].track_id == 3

    def test_no_double_count_same_vehicle(self, mock_heavy_deps):
        """Un véhicule ne peut déclencher qu'une seule mesure même s'il reste visible."""
        mock_yolo, _ = mock_heavy_deps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=10.0)

        # L1 crossing
        _setup_yolo_track(mock_yolo, [_mock_box(100, 190, 200, 215, track_id=9)])
        d._process_frame(frame, 10, 25.0)

        # L2 crossing → résultat créé
        _setup_yolo_track(mock_yolo, [_mock_box(100, 388, 200, 415, track_id=9)])
        d._process_frame(frame, 28, 25.0)
        assert len(d._results) == 1

        # Encore près de L2 → pas de nouveau résultat
        _setup_yolo_track(mock_yolo, [_mock_box(100, 390, 200, 412, track_id=9)])
        d._process_frame(frame, 30, 25.0)
        assert len(d._results) == 1  # toujours 1


# ---------------------------------------------------------------------------
# Tests de _save_crop
# ---------------------------------------------------------------------------

class TestSaveCrop:
    def test_no_captures_dir_returns_empty(self, mock_heavy_deps):
        """Sans captures_dir, _save_crop retourne toujours ""."""
        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = d._save_crop(frame, (100, 100, 200, 200), 1, 50.0, "car", "12:00:00")
        assert result == ""

    def test_empty_crop_returns_empty(self, tmp_path, mock_heavy_deps):
        """Crop hors-image (taille zéro) → ""."""
        from src.speed_detector import SpeedDetector
        d = SpeedDetector(
            line1_y=200, line2_y=400, real_distance_m=8.0,
            captures_dir=str(tmp_path / "caps"),
        )
        # Box complètement en dessous du frame 100×100 → slice vide
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = d._save_crop(frame, (0, 200, 50, 300), 1, 50.0, "car", "12:00:00")
        assert result == ""

    def test_valid_box_returns_path_string(self, tmp_path, mock_heavy_deps):
        """Crop valide → chemin de fichier retourné (cv2.imwrite mocké)."""
        from src.speed_detector import SpeedDetector
        caps_dir = tmp_path / "caps"
        d = SpeedDetector(
            line1_y=200, line2_y=400, real_distance_m=8.0,
            captures_dir=str(caps_dir),
        )
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = d._save_crop(frame, (50, 100, 200, 300), 1, 52.0, "car", "14:30:00")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests de _read_plate sans lecteur
# ---------------------------------------------------------------------------

class TestReadPlate:
    def test_no_plate_reader_returns_empty(self, mock_heavy_deps):
        """Sans captures_dir, _plate_reader est None → retourne ""."""
        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        assert d._plate_reader is None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert d._read_plate(frame, (100, 200, 200, 300)) == ""


# ---------------------------------------------------------------------------
# Tests CSV — cas limite
# ---------------------------------------------------------------------------

class TestCsvEdgeCases:
    def test_write_csv_empty_results(self, tmp_path, mock_heavy_deps):
        """CSV sans résultats → fichier créé avec uniquement les headers."""
        from src.speed_detector import SpeedDetector
        d = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        csv_path = tmp_path / "empty.csv"
        d._write_csv(csv_path)

        assert csv_path.exists()
        with csv_path.open() as f:
            lines = f.readlines()
        assert len(lines) == 1  # seulement le header
        assert "speed_kmh" in lines[0]


# ---------------------------------------------------------------------------
# Tests calibrate_from_frame
# ---------------------------------------------------------------------------

class TestCalibrate:
    def test_missing_frame_raises(self, mock_heavy_deps):
        """calibrate_from_frame avec image inexistante → FileNotFoundError."""
        _, mock_cv2 = mock_heavy_deps
        mock_cv2.imread.return_value = None  # simule fichier introuvable

        from src.speed_detector import SpeedDetector
        with pytest.raises(FileNotFoundError, match="introuvable"):
            SpeedDetector.calibrate_from_frame("/tmp/__inexistant_frame_xyz__.jpg")
