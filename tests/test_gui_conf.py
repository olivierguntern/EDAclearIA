"""Tests pour les fonctions pures de src/gui.py (_conf_label, VideoWorker).

PyQt6 est mocké pour éviter toute dépendance graphique.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixture — mock PyQt6 complet
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_pyqt6():
    """Mock PyQt6 avant chaque test pour éviter toute dépendance graphique."""
    pyqt6 = MagicMock()
    core = MagicMock()
    gui = MagicMock()
    widgets = MagicMock()

    pyqt6.QtCore = core
    pyqt6.QtGui = gui
    pyqt6.QtWidgets = widgets

    # QObject doit être une vraie classe pour permettre l'héritage
    core.QObject = object
    core.QThread = object
    # Qt.Orientation.Horizontal etc. comme MagicMock → acceptable

    patches = {
        "PyQt6": pyqt6,
        "PyQt6.QtCore": core,
        "PyQt6.QtGui": gui,
        "PyQt6.QtWidgets": widgets,
    }
    with patch.dict(sys.modules, patches):
        for mod in list(sys.modules):
            if mod == "src.gui":
                del sys.modules[mod]
        yield core, gui, widgets

    for mod in list(sys.modules):
        if mod == "src.gui":
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# Tests de _conf_label — logique pure, testée sans Qt
# ---------------------------------------------------------------------------

class TestConfLabel:
    def test_faible_minimum(self):
        from src.gui import _conf_label
        label = _conf_label(10)
        assert "Faible" in label
        assert "0.10" in label

    def test_faible_upper_boundary(self):
        from src.gui import _conf_label
        assert "Faible" in _conf_label(29)

    def test_normale_lower_boundary(self):
        from src.gui import _conf_label
        assert "Normale" in _conf_label(30)

    def test_normale_default(self):
        from src.gui import _conf_label
        label = _conf_label(40)
        assert "Normale" in label
        assert "0.40" in label

    def test_normale_upper_boundary(self):
        from src.gui import _conf_label
        assert "Normale" in _conf_label(55)

    def test_elevee_lower_boundary(self):
        from src.gui import _conf_label
        assert "Élevée" in _conf_label(56)

    def test_elevee_upper_boundary(self):
        from src.gui import _conf_label
        assert "Élevée" in _conf_label(75)

    def test_maximale_lower_boundary(self):
        from src.gui import _conf_label
        assert "Maximale" in _conf_label(76)

    def test_maximale_maximum(self):
        from src.gui import _conf_label
        label = _conf_label(90)
        assert "Maximale" in label
        assert "0.90" in label

    def test_value_always_present(self):
        """La valeur numérique (0.xx) est toujours incluse dans le label."""
        from src.gui import _conf_label
        for val in (10, 25, 40, 60, 80, 90):
            label = _conf_label(val)
            expected = f"{val / 100:.2f}"
            assert expected in label, f"valeur {expected} absente de '{label}'"

    def test_format_contains_parentheses(self):
        from src.gui import _conf_label
        label = _conf_label(40)
        assert "(" in label and ")" in label


# ---------------------------------------------------------------------------
# Tests VideoWorker — init et paramètres
# ---------------------------------------------------------------------------

class TestVideoWorkerInit:
    def test_parameters_stored(self):
        from src.gui import VideoWorker
        w = VideoWorker(
            video_path="test.mp4",
            line1_y=200,
            line2_y=400,
            real_distance_m=8.0,
            model_path="yolov8n.pt",
            speed_limit_kmh=50.0,
            conf=0.4,
        )
        assert w._video_path == "test.mp4"
        assert w._line1_y == 200
        assert w._line2_y == 400
        assert w._real_distance_m == 8.0
        assert w._model_path == "yolov8n.pt"
        assert w._speed_limit_kmh == 50.0
        assert w._conf == 0.4
        assert w._output_path is None
        assert w._captures_dir is None

    def test_optional_output_path(self):
        from src.gui import VideoWorker
        w = VideoWorker(
            video_path="v.mp4",
            line1_y=300,
            line2_y=450,
            real_distance_m=10.0,
            model_path="yolov8s.pt",
            speed_limit_kmh=0.0,
            conf=0.5,
            output_path="sortie.mp4",
            captures_dir="/tmp/caps",
        )
        assert w._output_path == "sortie.mp4"
        assert w._captures_dir == "/tmp/caps"

    def test_initial_display_time_is_zero(self):
        from src.gui import VideoWorker
        w = VideoWorker("v.mp4", 200, 400, 8.0, "yolov8n.pt", 50.0, 0.4)
        assert w._last_display_time == 0.0

    def test_detector_initially_none(self):
        from src.gui import VideoWorker
        w = VideoWorker("v.mp4", 200, 400, 8.0, "yolov8n.pt", 50.0, 0.4)
        assert w._detector is None


# ---------------------------------------------------------------------------
# Tests VideoWorker — throttling d'affichage
# ---------------------------------------------------------------------------

class TestVideoWorkerThrottle:
    def test_frame_throttled_below_interval(self, mock_pyqt6):
        """Deux frames consécutives très rapprochées → seule la première est émise."""
        from src.gui import VideoWorker
        w = VideoWorker("v.mp4", 200, 400, 8.0, "yolov8n.pt", 50.0, 0.4)
        w._last_display_time = time.monotonic()  # simule une frame récente

        emitted = []
        w.progress_updated = MagicMock()
        w.progress_updated.emit = lambda i, t: emitted.append(i)
        w.frame_ready = MagicMock()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        w._on_frame(frame, idx=5, total=100)

        # La frame throttled n'appelle pas frame_ready.emit
        w.frame_ready.emit.assert_not_called()

    def test_frame_emitted_after_interval(self, mock_pyqt6):
        """Après l'intervalle minimum, la frame est bien émise."""
        from src.gui import VideoWorker
        w = VideoWorker("v.mp4", 200, 400, 8.0, "yolov8n.pt", 50.0, 0.4)
        # Force _last_display_time très ancien
        w._last_display_time = 0.0

        w.frame_ready = MagicMock()
        w.progress_updated = MagicMock()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        w._on_frame(frame, idx=1, total=100)

        w.frame_ready.emit.assert_called_once()
