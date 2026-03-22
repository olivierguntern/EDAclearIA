"""Interface graphique PyQt6 pour la détection de vitesse en temps réel.

Lancement :
    python -m src.main --gui
    # ou directement :
    python -m src.gui
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PyQt6.QtCore import (
        QObject, QThread, Qt, QTimer, pyqtSignal,
    )
    from PyQt6.QtGui import QColor, QFont, QImage, QPalette, QPixmap
    from PyQt6.QtWidgets import (
        QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
        QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton,
        QSizePolicy, QSpinBox, QSplitter, QStatusBar, QTableWidget,
        QTableWidgetItem, QToolBar, QVBoxLayout, QWidget,
    )
except ImportError as exc:
    raise ImportError("PyQt6 est requis : pip install PyQt6") from exc


# ---------------------------------------------------------------------------
# Worker thread — fait tourner SpeedDetector sans bloquer l'UI
# ---------------------------------------------------------------------------

class VideoWorker(QObject):
    """Tourne dans un QThread et émet des signaux vers le thread principal."""

    frame_ready = pyqtSignal(QImage)                              # frame annotée (BGR→RGB)
    result_ready = pyqtSignal(int, float, float, str, str, str, str)  # (track_id, speed_kmh, timestamp_s, vehicle_type, wall_time, photo_path, plate_text)
    progress_updated = pyqtSignal(int, int)                       # (frame_idx, total_frames)
    finished = pyqtSignal(int)                                    # nb de mesures au total
    error_occurred = pyqtSignal(str)

    # Intervalle minimum entre deux frames affichées (s) → 30 fps max
    _DISPLAY_INTERVAL = 1 / 30

    def __init__(
        self,
        video_path: str,
        line1_y: int,
        line2_y: int,
        real_distance_m: float,
        model_path: str,
        speed_limit_kmh: float,
        conf: float,
        output_path: Optional[str] = None,
        captures_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._line1_y = line1_y
        self._line2_y = line2_y
        self._real_distance_m = real_distance_m
        self._model_path = model_path
        self._speed_limit_kmh = speed_limit_kmh
        self._conf = conf
        self._output_path = output_path
        self._captures_dir = captures_dir
        self._detector = None
        self._last_display_time = 0.0

    # ---- Slots appelés depuis le thread principal ----

    def stop(self) -> None:
        if self._detector:
            self._detector.stop()

    def pause(self) -> None:
        if self._detector:
            self._detector.pause()

    def resume(self) -> None:
        if self._detector:
            self._detector.resume()

    # ---- Entrée du thread ----

    def run(self) -> None:
        try:
            from src.speed_detector import SpeedDetector  # noqa: PLC0415

            self._detector = SpeedDetector(
                line1_y=self._line1_y,
                line2_y=self._line2_y,
                real_distance_m=self._real_distance_m,
                model_path=self._model_path,
                speed_limit_kmh=self._speed_limit_kmh,
                conf=self._conf,
                display=False,
                captures_dir=self._captures_dir,
            )

            results = self._detector.process_video(
                input_path=self._video_path,
                output=self._output_path,
                frame_callback=self._on_frame,
                result_callback=self._on_result,
            )
            self.finished.emit(len(results))

        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))

    # ---- Callbacks (appelés depuis le thread de traitement) ----

    def _on_frame(self, bgr: np.ndarray, idx: int, total: int) -> None:
        # Throttle display à 30 fps
        now = time.monotonic()
        if now - self._last_display_time < self._DISPLAY_INTERVAL:
            # Émet quand même la progression
            self.progress_updated.emit(idx, total)
            return
        self._last_display_time = now

        rgb = bgr[:, :, ::-1].copy()  # BGR → RGB, copie contiguë
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.frame_ready.emit(qimg.copy())  # copie pour thread-safety
        self.progress_updated.emit(idx, total)

    def _on_result(self, result) -> None:
        self.result_ready.emit(
            result.track_id,
            result.speed_kmh,
            result.timestamp_s,
            result.vehicle_type,
            result.wall_time,
            result.photo_path,
            result.plate_text,
        )


# ---------------------------------------------------------------------------
# Fenêtre principale
# ---------------------------------------------------------------------------

class SpeedDetectionWindow(QMainWindow):
    """Fenêtre principale de détection de vitesse."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Détection de vitesse — EDAclearIA")
        self.resize(1280, 780)

        self._worker: Optional[VideoWorker] = None
        self._thread: Optional[QThread] = None
        self._is_paused = False
        self._speeds: list[float] = []
        self._alert_count = 0
        self._output_path: Optional[str] = None
        self._captures_dir: Optional[str] = None
        self._row_photos: list[str] = []  # photo_path par ligne du tableau

        self._build_ui()
        self._connect_actions()

    # ------------------------------------------------------------------
    # Construction de l'UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- Toolbar ----
        tb = QToolBar("Contrôles")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._btn_open = QPushButton("Ouvrir vidéo")
        self._btn_open.setToolTip("Choisir un fichier vidéo")
        self._btn_start = QPushButton("Lancer")
        self._btn_start.setEnabled(False)
        self._btn_pause = QPushButton("Pause")
        self._btn_pause.setEnabled(False)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_export = QPushButton("Exporter CSV")
        self._btn_export.setEnabled(False)
        self._btn_save_video = QPushButton("Sauver vidéo")
        self._btn_save_video.setToolTip("Activer l'enregistrement de la vidéo annotée")
        self._btn_save_video.setCheckable(True)

        self._btn_captures = QPushButton("📷 Captures")
        self._btn_captures.setToolTip("Activer la sauvegarde des photos véhicules + lecture de plaque")
        self._btn_captures.setCheckable(True)

        for btn in (
            self._btn_open, self._btn_start, self._btn_pause,
            self._btn_stop, self._btn_export, self._btn_save_video, self._btn_captures,
        ):
            tb.addWidget(btn)

        tb.addSeparator()
        tb.addWidget(QLabel("  Modèle : "))
        self._combo_model = QComboBox()
        self._combo_model.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
        self._combo_model.setToolTip("Nano = rapide, Large = précis")
        tb.addWidget(self._combo_model)

        # ---- Corps principal (splitter gauche/droite) ----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # --- Panneau gauche : vidéo ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self._video_label = QLabel("Ouvrez une vidéo pour commencer")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_label.setStyleSheet("background: #1a1a2e; color: #aaa; border-radius: 4px;")
        self._video_label.setMinimumSize(640, 400)
        left_layout.addWidget(self._video_label)

        # Barre de progression
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v / %m frames (%p%)")
        left_layout.addWidget(self._progress)

        # Paramètres (ligne sous la vidéo)
        params_box = QGroupBox("Paramètres de calibration")
        params_layout = QHBoxLayout(params_box)

        params_layout.addWidget(QLabel("Ligne 1 (px) :"))
        self._spin_line1 = QSpinBox()
        self._spin_line1.setRange(0, 9999)
        self._spin_line1.setValue(300)
        self._spin_line1.setToolTip("Ordonnée Y de la première ligne virtuelle (pixels)")
        params_layout.addWidget(self._spin_line1)

        params_layout.addWidget(QLabel("Ligne 2 (px) :"))
        self._spin_line2 = QSpinBox()
        self._spin_line2.setRange(0, 9999)
        self._spin_line2.setValue(450)
        self._spin_line2.setToolTip("Ordonnée Y de la deuxième ligne virtuelle (pixels)")
        params_layout.addWidget(self._spin_line2)

        params_layout.addWidget(QLabel("Distance réelle (m) :"))
        self._spin_dist = QDoubleSpinBox()
        self._spin_dist.setRange(0.1, 999.0)
        self._spin_dist.setValue(8.0)
        self._spin_dist.setSingleStep(0.5)
        self._spin_dist.setDecimals(1)
        self._spin_dist.setToolTip("Distance au sol entre les deux lignes (mètres)")
        params_layout.addWidget(self._spin_dist)

        params_layout.addWidget(QLabel("Limite (km/h) :"))
        self._spin_limit = QDoubleSpinBox()
        self._spin_limit.setRange(0, 300)
        self._spin_limit.setValue(50.0)
        self._spin_limit.setDecimals(0)
        self._spin_limit.setToolTip("Seuil d'alerte vitesse (0 = désactivé)")
        params_layout.addWidget(self._spin_limit)

        params_layout.addWidget(QLabel("Confiance :"))
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.1, 1.0)
        self._spin_conf.setValue(0.4)
        self._spin_conf.setSingleStep(0.05)
        self._spin_conf.setDecimals(2)
        params_layout.addWidget(self._spin_conf)

        params_layout.addStretch()
        left_layout.addWidget(params_box)

        splitter.addWidget(left)

        # --- Panneau droit : stats + historique ----
        right = QWidget()
        right.setMaximumWidth(460)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Compteur vitesse en cours (grand affichage)
        self._lbl_last_speed = QLabel("-- km/h")
        font_big = QFont("Consolas", 36, QFont.Weight.Bold)
        self._lbl_last_speed.setFont(font_big)
        self._lbl_last_speed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_last_speed.setStyleSheet(
            "color: #00e676; background: #0d0d1a; border-radius: 8px; padding: 12px;"
        )
        right_layout.addWidget(self._lbl_last_speed)

        # Stats groupbox
        stats_box = QGroupBox("Statistiques")
        stats_layout = QVBoxLayout(stats_box)
        self._lbl_count = self._make_stat_label("Véhicules mesurés", "0")
        self._lbl_max = self._make_stat_label("Vitesse max", "-- km/h")
        self._lbl_min = self._make_stat_label("Vitesse min", "-- km/h")
        self._lbl_avg = self._make_stat_label("Vitesse moyenne", "-- km/h")
        self._lbl_alerts = self._make_stat_label("Dépassements", "0")
        for row in (self._lbl_count, self._lbl_max, self._lbl_min, self._lbl_avg, self._lbl_alerts):
            stats_layout.addLayout(row)
        right_layout.addWidget(stats_box)

        # Aperçu photo véhicule (affiché au clic d'une ligne)
        photo_box = QGroupBox("Photo véhicule")
        photo_layout = QVBoxLayout(photo_box)
        self._photo_preview = QLabel("Cliquez une ligne pour voir la photo")
        self._photo_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._photo_preview.setFixedHeight(160)
        self._photo_preview.setStyleSheet(
            "background: #0d0d1a; color: #555; border-radius: 4px; font-size: 11px;"
        )
        photo_layout.addWidget(self._photo_preview)
        right_layout.addWidget(photo_box)

        # Historique des mesures
        hist_box = QGroupBox("Historique des mesures")
        hist_layout = QVBoxLayout(hist_box)
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(["N°", "ID", "Type", "Heure", "km/h", "Plaque", "Alerte"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 30)
        self._table.setColumnWidth(1, 35)
        self._table.setColumnWidth(2, 58)
        self._table.setColumnWidth(3, 68)
        self._table.setColumnWidth(4, 50)
        self._table.setColumnWidth(5, 88)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        hist_layout.addWidget(self._table)
        right_layout.addWidget(hist_box, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([820, 460])

        # ---- Status bar ----
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._lbl_video_path = QLabel("Aucune vidéo sélectionnée")
        self._status.addWidget(self._lbl_video_path)

    @staticmethod
    def _make_stat_label(title: str, value: str) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl_title = QLabel(f"{title} :")
        lbl_title.setStyleSheet("color: #888;")
        lbl_val = QLabel(value)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        lbl_val.setStyleSheet("font-weight: bold;")
        row.addWidget(lbl_title)
        row.addWidget(lbl_val)
        # Expose the value label so we can update it later
        row._value_label = lbl_val  # type: ignore[attr-defined]
        return row

    # ------------------------------------------------------------------
    # Connexions signaux/slots
    # ------------------------------------------------------------------

    def _connect_actions(self) -> None:
        self._btn_open.clicked.connect(self._open_video)
        self._btn_start.clicked.connect(self._start)
        self._btn_pause.clicked.connect(self._toggle_pause)
        self._btn_stop.clicked.connect(self._stop)
        self._btn_export.clicked.connect(self._export_csv)
        self._btn_save_video.toggled.connect(self._toggle_save_video)
        self._btn_captures.toggled.connect(self._toggle_captures)
        self._table.itemSelectionChanged.connect(self._on_table_selection)

    # ------------------------------------------------------------------
    # Actions utilisateur
    # ------------------------------------------------------------------

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une vidéo", "",
            "Vidéos (*.mp4 *.avi *.mov *.mkv *.webm);;Tous les fichiers (*)"
        )
        if path:
            self._video_path = path
            self._lbl_video_path.setText(f"Vidéo : {Path(path).name}")
            self._btn_start.setEnabled(True)
            # Affiche la première frame comme aperçu
            self._show_first_frame(path)
            self._status.showMessage(f"Vidéo chargée : {path}", 3000)

    def _show_first_frame(self, path: str) -> None:
        import cv2  # noqa: PLC0415
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            self._display_frame(frame)

    def _start(self) -> None:
        if not hasattr(self, "_video_path"):
            return

        self._speeds.clear()
        self._alert_count = 0
        self._table.setRowCount(0)
        self._update_stats()
        self._lbl_last_speed.setText("-- km/h")
        self._lbl_last_speed.setStyleSheet(
            "color: #00e676; background: #0d0d1a; border-radius: 8px; padding: 12px;"
        )
        self._progress.setValue(0)
        self._btn_export.setEnabled(False)

        self._row_photos.clear()

        self._worker = VideoWorker(
            video_path=self._video_path,
            line1_y=self._spin_line1.value(),
            line2_y=self._spin_line2.value(),
            real_distance_m=self._spin_dist.value(),
            model_path=self._combo_model.currentText(),
            speed_limit_kmh=self._spin_limit.value(),
            conf=self._spin_conf.value(),
            output_path=self._output_path,
            captures_dir=self._captures_dir,
        )

        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.result_ready.connect(self._on_result)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error_occurred.connect(self._thread.quit)

        self._thread.start()

        self._btn_start.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._btn_open.setEnabled(False)
        self._is_paused = False
        self._status.showMessage("Traitement en cours…")

    def _toggle_pause(self) -> None:
        if not self._worker:
            return
        if self._is_paused:
            self._worker.resume()
            self._btn_pause.setText("Pause")
            self._is_paused = False
            self._status.showMessage("Traitement repris.")
        else:
            self._worker.pause()
            self._btn_pause.setText("Reprendre")
            self._is_paused = True
            self._status.showMessage("Traitement en pause.")

    def _stop(self) -> None:
        if self._worker:
            self._worker.stop()
        self._status.showMessage("Arrêt demandé…")

    def _toggle_save_video(self, checked: bool) -> None:
        if checked:
            path, _ = QFileDialog.getSaveFileName(
                self, "Sauvegarder la vidéo annotée", "sortie_vitesses.mp4",
                "Vidéos MP4 (*.mp4)"
            )
            if path:
                self._output_path = path
                self._btn_save_video.setText(f"Sauver : {Path(path).name}")
                self._status.showMessage(f"Sortie vidéo : {path}", 3000)
            else:
                self._btn_save_video.setChecked(False)
                self._output_path = None
        else:
            self._output_path = None
            self._btn_save_video.setText("Sauver vidéo")

    def _toggle_captures(self, checked: bool) -> None:
        if checked:
            if not hasattr(self, "_video_path"):
                self._btn_captures.setChecked(False)
                self._status.showMessage("Ouvrez d'abord une vidéo.", 3000)
                return
            from datetime import datetime as _dt  # noqa: PLC0415
            video_p = Path(self._video_path)
            session_ts = _dt.now().strftime("%Y-%m-%d_%H%M%S")
            captures_dir = video_p.parent / "captures" / f"{video_p.stem}_{session_ts}"
            self._captures_dir = str(captures_dir)
            self._btn_captures.setText(f"📷 {captures_dir.name}")
            self._status.showMessage(f"Captures → {captures_dir}", 4000)
        else:
            self._captures_dir = None
            self._btn_captures.setText("📷 Captures")

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter les résultats", "vitesses.csv", "CSV (*.csv)"
        )
        if not path:
            return

        import csv as csv_mod  # noqa: PLC0415
        fieldnames = ["no", "track_id", "type_vehicule", "heure_passage", "speed_kmh", "plaque", "photo_path", "alerte"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self._table.rowCount()):
                photo = self._row_photos[row] if row < len(self._row_photos) else ""
                writer.writerow({
                    "no": self._table.item(row, 0).text(),
                    "track_id": self._table.item(row, 1).text(),
                    "type_vehicule": self._table.item(row, 2).text(),
                    "heure_passage": self._table.item(row, 3).text(),
                    "speed_kmh": self._table.item(row, 4).text(),
                    "plaque": self._table.item(row, 5).text(),
                    "photo_path": photo,
                    "alerte": self._table.item(row, 6).text(),
                })
        self._status.showMessage(f"CSV exporté → {path}", 4000)

    # ------------------------------------------------------------------
    # Slots connectés au worker
    # ------------------------------------------------------------------

    def _on_frame(self, qimg: QImage) -> None:
        self._display_qimage(qimg)

    def _on_result(
        self,
        track_id: int,
        speed_kmh: float,
        timestamp_s: float,
        vehicle_type: str,
        wall_time: str,
        photo_path: str,
        plate_text: str,
    ) -> None:
        self._speeds.append(speed_kmh)
        self._row_photos.append(photo_path)
        limit = self._spin_limit.value()
        is_alert = limit > 0 and speed_kmh > limit
        if is_alert:
            self._alert_count += 1

        # Grand affichage vitesse
        self._lbl_last_speed.setText(f"{speed_kmh:.1f} km/h")
        if is_alert:
            self._lbl_last_speed.setStyleSheet(
                "color: #ff1744; background: #1a0a0a; border-radius: 8px; padding: 12px;"
            )
        else:
            self._lbl_last_speed.setStyleSheet(
                "color: #00e676; background: #0d0d1a; border-radius: 8px; padding: 12px;"
            )

        # Tableau historique — 7 colonnes : N°, ID, Type, Heure, km/h, Plaque, Alerte
        row = self._table.rowCount()
        self._table.insertRow(row)

        for col, text, align in [
            (0, str(row + 1), True),
            (1, str(track_id), True),
            (2, vehicle_type, True),
            (3, wall_time, True),
            (4, f"{speed_kmh:.1f}", True),
            (5, plate_text, True),
            (6, "⚠ OUI" if is_alert else "", True),
        ]:
            item = QTableWidgetItem(text)
            if align:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_alert and col in (4, 6):
                item.setForeground(QColor("#ff1744"))
            self._table.setItem(row, col, item)

        self._table.scrollToBottom()

        # Affiche la photo si disponible
        if photo_path:
            self._show_photo(photo_path)

        self._update_stats()

    def _on_table_selection(self) -> None:
        rows = self._table.selectedItems()
        if not rows:
            return
        row = self._table.currentRow()
        if 0 <= row < len(self._row_photos):
            photo_path = self._row_photos[row]
            if photo_path:
                self._show_photo(photo_path)
            else:
                self._photo_preview.setText("Captures non activées\npour cette session")

    def _show_photo(self, path: str) -> None:
        """Charge et affiche une photo véhicule dans l'aperçu."""
        from pathlib import Path as _Path  # noqa: PLC0415
        if not _Path(path).exists():
            self._photo_preview.setText(f"Photo introuvable :\n{_Path(path).name}")
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._photo_preview.setText("Impossible de charger la photo")
            return
        scaled = pixmap.scaled(
            self._photo_preview.width() - 8,
            self._photo_preview.height() - 8,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._photo_preview.setPixmap(scaled)

    def _on_progress(self, idx: int, total: int) -> None:
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(idx)

    def _on_finished(self, count: int) -> None:
        self._btn_start.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_open.setEnabled(True)
        self._btn_pause.setText("Pause")
        self._is_paused = False
        if count > 0:
            self._btn_export.setEnabled(True)
        self._status.showMessage(
            f"Terminé — {count} véhicule(s) mesuré(s)."
        )

    def _on_error(self, msg: str) -> None:
        self._btn_start.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_open.setEnabled(True)
        self._status.showMessage(f"Erreur : {msg}")
        self._video_label.setText(f"Erreur :\n{msg}")
        self._video_label.setStyleSheet(
            "background: #2a0a0a; color: #ff6b6b; border-radius: 4px;"
        )

    # ------------------------------------------------------------------
    # Helpers d'affichage
    # ------------------------------------------------------------------

    def _display_frame(self, bgr: np.ndarray) -> None:
        rgb = bgr[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._display_qimage(qimg)

    def _display_qimage(self, qimg: QImage) -> None:
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    def _update_stats(self) -> None:
        count = len(self._speeds)
        self._lbl_count._value_label.setText(str(count))  # type: ignore[attr-defined]
        if count:
            self._lbl_max._value_label.setText(f"{max(self._speeds):.1f} km/h")   # type: ignore[attr-defined]
            self._lbl_min._value_label.setText(f"{min(self._speeds):.1f} km/h")   # type: ignore[attr-defined]
            self._lbl_avg._value_label.setText(f"{sum(self._speeds)/count:.1f} km/h")  # type: ignore[attr-defined]
        self._lbl_alerts._value_label.setText(str(self._alert_count))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Fermeture propre
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if self._worker:
            self._worker.stop()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        event.accept()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def run_gui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    # Thème sombre via palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1a1a2e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#16213e"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#0f3460"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#0f3460"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#e94560"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyle("Fusion")

    window = SpeedDetectionWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
