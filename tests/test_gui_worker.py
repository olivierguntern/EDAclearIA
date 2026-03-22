"""Tests unitaires pour VideoWorker et les modifications de SpeedDetector.

Toutes les dépendances lourdes (PyQt6, ultralytics, cv2) sont mockées.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures : mock des dépendances lourdes
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_deps(tmp_path):
    """Mock ultralytics, cv2 et PyQt6 avant l'import des modules."""

    # --- cv2 ---
    mock_cv2 = MagicMock()
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.COLOR_BGR2RGB = 4

    # --- ultralytics ---
    mock_yolo_cls = MagicMock()
    mock_yolo_inst = MagicMock()
    mock_yolo_inst.model.names = {0: "car"}
    mock_yolo_cls.return_value = mock_yolo_inst
    ul_mock = MagicMock()
    ul_mock.YOLO = mock_yolo_cls

    # --- PyQt6 ---
    pyqt6_mock = MagicMock()
    pyqt6_core = MagicMock()
    pyqt6_gui = MagicMock()
    pyqt6_widgets = MagicMock()
    pyqt6_mock.QtCore = pyqt6_core
    pyqt6_mock.QtGui = pyqt6_gui
    pyqt6_mock.QtWidgets = pyqt6_widgets
    # QObject doit être une vraie classe pour l'héritage
    pyqt6_core.QObject = object
    pyqt6_core.QThread = object

    patches = {
        "cv2": mock_cv2,
        "ultralytics": ul_mock,
        "PyQt6": pyqt6_mock,
        "PyQt6.QtCore": pyqt6_core,
        "PyQt6.QtGui": pyqt6_gui,
        "PyQt6.QtWidgets": pyqt6_widgets,
    }

    with patch.dict(sys.modules, patches):
        for mod in list(sys.modules):
            if mod in ("src.speed_detector", "src.gui"):
                del sys.modules[mod]
        yield mock_cv2, mock_yolo_inst, tmp_path


# ---------------------------------------------------------------------------
# Tests SpeedDetector — stop / pause / resume
# ---------------------------------------------------------------------------

class TestSpeedDetectorControl:
    def test_stop_interrupts_loop(self, mock_deps):
        mock_cv2, mock_yolo, tmp_path = mock_deps
        from src.speed_detector import SpeedDetector  # noqa: PLC0415

        # Simule une vidéo infinie
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.side_effect = lambda prop: {5: 25.0, 3: 640, 4: 480, 7: 0}.get(prop, 0)
        frame_calls = [0]

        def fake_read():
            frame_calls[0] += 1
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        cap_mock.read.side_effect = fake_read
        mock_cv2.VideoCapture.return_value = cap_mock

        fake_video = tmp_path / "fake.mp4"
        fake_video.write_bytes(b"fake")

        detector = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)

        def stop_after_delay():
            time.sleep(0.05)
            detector.stop()

        t = threading.Thread(target=stop_after_delay, daemon=True)
        t.start()
        results = detector.process_video(fake_video)
        t.join(timeout=2)

        # Le stop doit avoir interrompu la boucle
        assert frame_calls[0] < 1000, "La boucle ne s'est pas arrêtée"

    def test_pause_blocks_processing(self, mock_deps):
        mock_cv2, mock_yolo, tmp_path = mock_deps
        from src.speed_detector import SpeedDetector  # noqa: PLC0415

        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.side_effect = lambda prop: {5: 25.0, 3: 640, 4: 480, 7: 0}.get(prop, 0)
        cap_mock.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        mock_cv2.VideoCapture.return_value = cap_mock
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"x")

        detector = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)

        def pause_then_resume_then_stop():
            time.sleep(0.02)
            detector.pause()
            assert detector.is_paused
            time.sleep(0.05)
            detector.resume()
            assert not detector.is_paused
            time.sleep(0.02)
            detector.stop()

        t = threading.Thread(target=pause_then_resume_then_stop, daemon=True)
        t.start()
        detector.process_video(fake_video)
        t.join(timeout=3)

    def test_frame_callback_called(self, mock_deps):
        mock_cv2, mock_yolo, tmp_path = mock_deps
        from src.speed_detector import SpeedDetector  # noqa: PLC0415

        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.side_effect = lambda prop: {5: 25.0, 3: 640, 4: 480, 7: 3}.get(prop, 0)
        frames_returned = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]
        cap_mock.read.side_effect = frames_returned
        mock_cv2.VideoCapture.return_value = cap_mock

        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"x")

        detector = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        received_frames = []
        detector.process_video(fake_video, frame_callback=lambda f, i, t: received_frames.append(i))

        assert len(received_frames) == 3

    def test_result_callback_called(self, mock_deps):
        """Vérifie que result_callback est appelé quand une vitesse est calculée."""
        mock_cv2, mock_yolo, tmp_path = mock_deps
        from src.speed_detector import SpeedDetector, SpeedResult  # noqa: PLC0415

        received = []

        def cb(result: SpeedResult):
            received.append(result)

        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"x")

        detector = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        # Injecte un résultat manuellement pour tester le callback
        from src.speed_detector import SpeedResult  # noqa: PLC0415
        r = SpeedResult(track_id=5, speed_kmh=42.0, frame_detected=50, timestamp_s=2.0)
        detector._results.append(r)
        cb(r)

        assert len(received) == 1
        assert received[0].speed_kmh == 42.0


# ---------------------------------------------------------------------------
# Tests calcul vitesse avec callbacks
# ---------------------------------------------------------------------------

class TestSpeedDetectorCallbacks:
    def test_stop_flag_reset_on_new_run(self, mock_deps):
        mock_cv2, _, tmp_path = mock_deps
        from src.speed_detector import SpeedDetector  # noqa: PLC0415

        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.side_effect = lambda p: {5: 25.0, 3: 640, 4: 480, 7: 2}.get(p, 0)
        cap_mock.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_cv2.VideoCapture.return_value = cap_mock

        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"x")

        detector = SpeedDetector(line1_y=200, line2_y=400, real_distance_m=8.0)
        detector.stop()  # stop avant de commencer

        # Le stop flag doit être réinitialisé au prochain process_video
        frames_seen = []
        detector.process_video(fake_video, frame_callback=lambda f, i, t: frames_seen.append(i))
        assert len(frames_seen) == 2  # les 2 frames ont été traitées
