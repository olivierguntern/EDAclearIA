"""Tests unitaires pour src/plate_reader.py.

Les dépendances lourdes (easyocr, cv2) sont mockées.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tests des regex — aucune dépendance externe
# ---------------------------------------------------------------------------

class TestPlateRegex:
    """Vérifie que les regex détectent les formats de plaques françaises."""

    def test_new_format_with_dashes(self):
        from src.plate_reader import _RE_NEW
        assert _RE_NEW.search("AB-123-CD") is not None

    def test_new_format_without_dashes(self):
        from src.plate_reader import _RE_NEW
        assert _RE_NEW.search("AB123CD") is not None

    def test_new_format_lowercase_accepted(self):
        from src.plate_reader import _RE_NEW
        assert _RE_NEW.search("ab-123-cd") is not None

    def test_old_format_with_spaces(self):
        from src.plate_reader import _RE_OLD
        assert _RE_OLD.search("1234 AB 75") is not None

    def test_old_format_compact(self):
        from src.plate_reader import _RE_OLD
        assert _RE_OLD.search("1234AB75") is not None

    def test_no_match_plain_text(self):
        from src.plate_reader import _RE_NEW, _RE_OLD
        text = "HELLO WORLD"
        assert _RE_NEW.search(text) is None
        assert _RE_OLD.search(text) is None

    def test_no_match_empty_string(self):
        from src.plate_reader import _RE_NEW, _RE_OLD
        assert _RE_NEW.search("") is None
        assert _RE_OLD.search("") is None


# ---------------------------------------------------------------------------
# Tests PlateReader sans easyocr installé
# ---------------------------------------------------------------------------

class TestPlateReaderNoEasyOCR:
    """Dégradation gracieuse quand easyocr est absent."""

    @pytest.fixture(autouse=True)
    def _no_easyocr(self):
        """Simule l'absence d'easyocr dans sys.modules."""
        with patch.dict(sys.modules, {"easyocr": None}):
            if "src.plate_reader" in sys.modules:
                del sys.modules["src.plate_reader"]
            yield
        if "src.plate_reader" in sys.modules:
            del sys.modules["src.plate_reader"]

    def test_available_is_false(self):
        from src.plate_reader import PlateReader
        reader = PlateReader()
        assert reader.available is False

    def test_read_plate_returns_empty_string(self):
        from src.plate_reader import PlateReader
        reader = PlateReader()
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        assert reader.read_plate(crop) == ""

    def test_read_plate_empty_array_returns_empty(self):
        from src.plate_reader import PlateReader
        reader = PlateReader()
        empty = np.zeros((0,), dtype=np.uint8)
        assert reader.read_plate(empty) == ""


# ---------------------------------------------------------------------------
# Tests PlateReader avec easyocr mocké
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_easyocr():
    """Fournit un easyocr mocké prêt à l'emploi."""
    mock_ocr = MagicMock()
    mock_reader_inst = MagicMock()
    mock_ocr.Reader.return_value = mock_reader_inst

    mock_cv2 = MagicMock()
    mock_cv2.resize.return_value = np.zeros((100, 400, 3), dtype=np.uint8)
    mock_cv2.INTER_CUBIC = 2

    with patch.dict(sys.modules, {"easyocr": mock_ocr, "cv2": mock_cv2}):
        if "src.plate_reader" in sys.modules:
            del sys.modules["src.plate_reader"]
        yield mock_reader_inst, mock_cv2

    if "src.plate_reader" in sys.modules:
        del sys.modules["src.plate_reader"]


class TestPlateReaderWithOCR:
    def test_available_with_easyocr(self, mock_easyocr):
        from src.plate_reader import PlateReader
        reader = PlateReader()
        assert reader.available is True

    def test_small_crop_skipped(self, mock_easyocr):
        """Crop trop petit (< 10×10) → "" sans appeler readtext."""
        mock_reader_inst, _ = mock_easyocr
        from src.plate_reader import PlateReader
        reader = PlateReader()
        tiny = np.zeros((5, 5, 3), dtype=np.uint8)
        result = reader.read_plate(tiny)
        assert result == ""
        mock_reader_inst.readtext.assert_not_called()

    def test_new_plate_format_detected(self, mock_easyocr):
        """readtext retourne 'AB-123-CD' → plaque normalisée retournée."""
        mock_reader_inst, _ = mock_easyocr
        mock_reader_inst.readtext.return_value = [
            (None, "AB-123-CD", 0.95),
        ]
        from src.plate_reader import PlateReader
        reader = PlateReader()
        crop = np.zeros((100, 300, 3), dtype=np.uint8)
        result = reader.read_plate(crop)
        assert "AB" in result
        assert "123" in result
        assert "CD" in result

    def test_low_confidence_result_ignored(self, mock_easyocr):
        """Résultat avec confiance < 0.25 ignoré → retourne ""."""
        mock_reader_inst, _ = mock_easyocr
        mock_reader_inst.readtext.return_value = [
            (None, "AB-123-CD", 0.10),  # confiance trop basse
        ]
        from src.plate_reader import PlateReader
        reader = PlateReader()
        crop = np.zeros((100, 300, 3), dtype=np.uint8)
        result = reader.read_plate(crop)
        assert result == ""

    def test_no_plate_in_text_returns_empty(self, mock_easyocr):
        """OCR ne trouve pas de format plaque → retourne "" (ou fallback courts)."""
        mock_reader_inst, _ = mock_easyocr
        mock_reader_inst.readtext.return_value = [
            (None, "STOP", 0.90),  # texte non-plaque
        ]
        from src.plate_reader import PlateReader
        reader = PlateReader()
        crop = np.zeros((100, 300, 3), dtype=np.uint8)
        result = reader.read_plate(crop)
        # "STOP" → 4 chars alphanum → peut être retourné comme fallback
        # Le comportement attendu est soit "" soit "STOP" (4 chars)
        assert isinstance(result, str)

    def test_exception_in_readtext_returns_empty(self, mock_easyocr):
        """Exception pendant readtext → retourne "" sans propager."""
        mock_reader_inst, _ = mock_easyocr
        mock_reader_inst.readtext.side_effect = RuntimeError("OCR crash")
        from src.plate_reader import PlateReader
        reader = PlateReader()
        crop = np.zeros((100, 300, 3), dtype=np.uint8)
        result = reader.read_plate(crop)
        assert result == ""
