########### OLD UI ################

from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QSpinBox, QFormLayout, QProgressBar, QMessageBox, QCheckBox
)

from .solver import solve_string_art_go
from .previewer import simulate_string_art
from StringArtConverter import preprocessing

def to_qpixmap_from_rgb(rgb: np.ndarray, fit_size: tuple[int, int] | None = None) -> QPixmap:
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    pm = QPixmap.fromImage(qimg)
    if fit_size:
        pm = pm.scaled(fit_size[0], fit_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm

class ConvertWorker(QObject):
    progress = Signal(int)
    finished = Signal(list)   # list[(from_pin, to_pin)]
    errored = Signal(str)

    def __init__(self, img_bgr, n_pins: int, steps: int, min_hop: int, use_multi: bool = False, k_strings: int = 3):
        super().__init__()
        self.img_bgr = img_bgr
        self.n_pins = n_pins
        self.steps = steps
        self.min_hop = min_hop
        self.use_multi = use_multi
        self.k_strings = k_strings

    @Slot()
    def run(self):
        try:
            path = solve_string_art_go(
                self.img_bgr,
                self.n_pins,
                self.steps,
                self.min_hop,
                progress_cb=self.progress.emit,
            )
            self.finished.emit(path)
        except Exception as e:
            self.errored.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("String Art Converter")
        self.resize(1100, 720)

        self.img_bgr = None
        self.current_path: List[Tuple[int, int]] = []

        central = QWidget()
        self.setCentralWidget(central)

        self.image_label = QLabel("Drop an image here or use Open…")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border:1px dashed #888; border-radius:12px; padding:16px;")

        form = QFormLayout()
        # choose number of pins and steps
        self.spin_pins = QSpinBox()
        self.spin_pins.setRange(12, 1024)
        self.spin_pins.setValue(200)
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 5000)
        self.spin_steps.setValue(1500)
        # conversion progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        # checkbox for multi solver + how many strings
        self.chk_multi = QCheckBox("Use multiple strings")
        self.spin_strings = QSpinBox()
        self.spin_strings.setRange(2, 12)    
        self.spin_strings.setValue(3)
        self.spin_strings.setEnabled(False)   # only enabled when checkbox is checked
        self.chk_multi.toggled.connect(self.spin_strings.setEnabled)
        # Buttons
        self.btn_preview = QPushButton("Render Preview")
        self.btn_preview.setEnabled(False)
        self.btn_open = QPushButton("Open Image…")
        self.btn_convert = QPushButton("Convert to String Art")
        self.btn_convert.setEnabled(False)

        form.addRow("Pins (circle):", self.spin_pins)
        form.addRow("Steps:", self.spin_steps)
        form.addRow(self.btn_open)
        form.addRow(self.btn_convert)
        form.addRow("Progress:", self.progress)
        form.addRow(self.btn_preview)
        form.addRow(self.chk_multi)
        form.addRow("Strings (K):", self.spin_strings)

        right = QWidget(); right.setLayout(form)
        root = QHBoxLayout(central)
        root.addWidget(self.image_label, 1)
        root.addWidget(right, 0)

        export_csv = QAction("Export coordinates (CSV)…", self)
        export_csv.triggered.connect(self.export_csv)
        self.menuBar().addMenu("&File").addAction(export_csv)

        self.btn_open.clicked.connect(self.open_image)
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_preview.clicked.connect(self.render_preview)

        self.setAcceptDrops(True)
        self._thread: QThread | None = None
        self._worker: ConvertWorker | None = None

    def dragEnterEvent(self, e): 
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            self.load_image(url.toLocalFile()); break

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path: self.load_image(path)

    def load_image(self, path: str):
        bgr = cv2.imread(path)
        if bgr is None:
            QMessageBox.warning(self, "Open image", "Could not read the image."); return
        self.img_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.image_label.setPixmap(to_qpixmap_from_rgb(rgb, (self.image_label.width(), self.image_label.height())))
        self.btn_convert.setEnabled(True)
        self.progress.setValue(0)
        self.current_path = []
        self.btn_preview.setEnabled(bool(self.current_path))
        self.btn_preview.setEnabled(False)

    def start_conversion(self):
        if self.img_bgr is None:
            QMessageBox.information(self, "No image", "Load an image first."); return

        # Preprocessing preview,  is done in solver already
        preview_rgb = preprocessing.preview(self.img_bgr, edge_weight=0.5, max_size=512)
        h, w, _ = preview_rgb.shape
        qimg = QImage(preview_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        n_pins = self.spin_pins.value()
        steps = self.spin_steps.value()
        min_hop = max(6, n_pins // 40)
        use_multi = self.chk_multi.isChecked()
        k_strings = self.spin_strings.value()

        if self._thread:
            self._thread.quit(); self._thread.wait()

        self._thread = QThread()
        self._worker = ConvertWorker(
                                    self.img_bgr, n_pins, steps, min_hop, 
                                    use_multi=use_multi, k_strings=k_strings
                                    )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(self.on_finished)
        self._worker.errored.connect(self.on_errored)
        self._worker.finished.connect(self._thread.quit)
        self._worker.errored.connect(self._thread.quit)

        self._thread.start()
        self.btn_convert.setEnabled(False); self.btn_open.setEnabled(False)

    def on_finished(self, path: List[Tuple[int, int]]):
        self.current_path = path
        self.btn_convert.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.btn_preview.setEnabled(True)
        if not path:
            QMessageBox.information(self, "Done", "No path produced (try fewer pins or more steps)."); return
        QMessageBox.information(self, "Conversion complete",
            f"Generated {len(path)} string segments.\nFirst 5: {path[:5]}")

    def on_errored(self, msg: str):
        self.btn_convert.setEnabled(True); self.btn_open.setEnabled(True)
        QMessageBox.critical(self, "Error during conversion", msg)

    def export_csv(self):
        if not self.current_path:
            QMessageBox.information(self, "Nothing to export", "Run a conversion first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save coordinates", "string_art.csv", "CSV (*.csv)")
        if not path: return
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["from_pin", "to_pin"]); w.writerows(self.current_path)
            QMessageBox.information(self, "Saved", f"Coordinates saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def render_preview(self):
        if not self.current_path:
            QMessageBox.information(self, "No path", "Run a conversion first to generate coordinates.")
            return

        # Size: render to the display label size for speed; you can choose a larger export size too
        width  = max(64, self.image_label.width())
        height = max(64, self.image_label.height())
        n_pins = self.spin_pins.value()

        # Simulate
        preview_rgb = simulate_string_art(
            width=width,
            height=height,
            n_pins=n_pins,
            segments=self.current_path,
            margin=16,
            background_color=(245, 245, 245),
            thread_color=(30, 30, 30),
            thickness=1,
            alpha_per_segment=0.08,
        )

        # Show it
        h, w, _ = preview_rgb.shape
        qimg = QImage(preview_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))