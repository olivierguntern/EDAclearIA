"""Lecture de plaque d'immatriculation par OCR (easyocr).

Dégradation gracieuse : si easyocr n'est pas installé,
PlateReader.read_plate() retourne toujours "" sans lever d'exception.

Installation optionnelle :
    pip install easyocr
"""

from __future__ import annotations

import re

import numpy as np

# Formats de plaques françaises
_RE_NEW = re.compile(r"[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{2}", re.IGNORECASE)   # AB-123-CD
_RE_OLD = re.compile(r"\d{1,4}[-\s]?[A-Z]{1,3}[-\s]?\d{2,3}", re.IGNORECASE)  # 1234 AB 75


class PlateReader:
    """Lecteur OCR de plaques d'immatriculation basé sur easyocr.

    Dégradation gracieuse : si easyocr n'est pas installé, read_plate()
    retourne toujours "" sans lever d'exception.
    """

    def __init__(self) -> None:
        self._reader = None
        try:
            import easyocr  # noqa: PLC0415
            self._reader = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)
        except ImportError:
            pass  # graceful degradation — plates column will be empty

    @property
    def available(self) -> bool:
        """True si easyocr est installé et le lecteur initialisé."""
        return self._reader is not None

    def read_plate(self, vehicle_crop: np.ndarray) -> str:
        """Tente d'extraire une plaque depuis un crop de véhicule.

        Stratégie :
        1. Isole la moitié basse du crop (où se trouve la plaque)
        2. Agrandit si nécessaire pour améliorer l'OCR
        3. Cherche un format de plaque connu (regex)
        4. Retourne la valeur brute si plausible (4–9 chars alphanum)

        Args:
            vehicle_crop: Image BGR du véhicule (bounding box découpée).

        Returns:
            Texte de la plaque normalisé (ex. "AB-123-CD") ou "" si non trouvé.
        """
        if self._reader is None or vehicle_crop.size == 0:
            return ""

        try:
            import cv2  # noqa: PLC0415

            h, w = vehicle_crop.shape[:2]
            if h < 10 or w < 10:
                return ""

            # Zone de recherche : moitié basse du véhicule (plaque avant/arrière)
            plate_zone = vehicle_crop[h // 2:, :]

            # Agrandissement si la zone est petite (améliore l'OCR)
            ph, pw = plate_zone.shape[:2]
            if pw < 200:
                scale = max(2, 200 // max(pw, 1))
                plate_zone = cv2.resize(
                    plate_zone, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            # OCR avec filtrage alphanumérique
            results = self._reader.readtext(
                plate_zone,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                detail=1,
            )

            # Filtre par confiance et concatène
            texts = [r[1].upper() for r in results if r[2] >= 0.25]
            full_text = " ".join(texts)

            # Cherche un format de plaque connu
            match = _RE_NEW.search(full_text) or _RE_OLD.search(full_text)
            if match:
                plate = match.group(0).upper()
                return re.sub(r"[-\s]+", "-", plate)

            # Fallback : retourne le texte brut si plausible (4–9 chars alphanum)
            clean = re.sub(r"[^A-Z0-9]", "", full_text)
            if 4 <= len(clean) <= 9:
                return clean

            return ""

        except Exception:  # noqa: BLE001
            return ""
