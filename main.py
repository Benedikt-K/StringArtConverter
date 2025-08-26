from __future__ import annotations
from typing import List, Optional
import sys, math
import cv2
import numpy as np
import json, os

from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize
from PySide6.QtGui import QAction, QPixmap, QImage, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QProgressBar, QMessageBox, QScrollArea, QFrame, QComboBox, QHBoxLayout
)

# -------- solver imports --------
from StringArtConverter.preprocessing import build_brightness_for_go_solver
from StringArtConverter.utils import save_path_txt, set_widget_ranges, load_presets_json, clamp_to_ranges, apply_to_widgets, collect_from_widgets, Segment
from StringArtConverter.previewer import render_path
from StringArtConverter.solver import solve_string_art_go


# ── small util helpers ───────────────────────────────────────────────────────
def to_qpixmap_from_rgb(rgb: np.ndarray, fit_size: Optional[QSize] = None) -> QPixmap:
    if rgb.ndim == 2:
        rgb = np.stack([rgb]*3, axis=-1)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    pm = QPixmap.fromImage(qimg)
    if fit_size is not None:
        pm = pm.scaled(fit_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm

def bgr_to_rgb_img(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def info(self, title, text):
    QMessageBox.information(self, title, text)

def warn(self, title, text):
    QMessageBox.warning(self, title, text)

def error(self, title, text):
    QMessageBox.critical(self, title, text)

# ── worker that runs in background thread ─────────────────────────────────────
class ConvertWorker(QObject):
    progress = Signal(int)
    finished = Signal(list, np.ndarray, np.ndarray, np.ndarray)  # path, error, target, pins
    errored = Signal(str)

    def __init__(self, img_bgr: np.ndarray, params: dict):
        super().__init__()
        self.img_bgr = img_bgr
        self.params = params

    def run(self):
        try:
            # preprocessing → brightness map (uint8 HxW)
            src_u8 = build_brightness_for_go_solver(
                img_bgr=self.img_bgr,
                work_size=self.params["work_size"],
                use_clahe=self.params["pp_clahe"],
                use_contrast=self.params["pp_contrast"],
                p_low=self.params["pp_c_low"],
                p_high=self.params["pp_c_high"],
                use_edges=self.params["pp_edges"],
                edge_weight=self.params["pp_edge_weight"],
                edge_low=self.params["pp_edge_low"],
                edge_high=self.params["pp_edge_high"],
                edge_auto_sigma=self.params["pp_edge_auto_sigma"],
                use_rembg=self.params["pp_rembg"],
                rembg_dim=self.params["pp_rembg_dim"],
                rembg_feather=self.params["pp_rembg_feather"],
                rembg_erode=self.params["pp_rembg_erode"],
                # optional: gamma + clipping if you added in your go_main:
                pp_gamma=self.params.get("pp_gamma", 1.0),
                pp_clip_high=self.params.get("pp_clip_high", 100.0),
            )

            def cb(p: int):
                self.progress.emit(p)

            path, err, target, pins = solve_string_art_go(
                source_brightness_u8=src_u8,
                n_pins=self.params["pins"],
                max_lines=self.params["steps"],
                min_distance=self.params["min_distance"],
                line_weight=self.params["line_weight"],
                last_n=self.params["last_n"],
                work_size=self.params["work_size"],
                progress_cb=cb,
            )
            self.finished.emit(path, err, target, pins)
        except Exception as e:
            self.errored.emit(str(e))

# ── App style ─────────────────────────────────────────────────────────
APP_STYLES = """
/* Global base */
QMainWindow {
    background: #0b0c10;
    color: #e6edf3;
}
QWidget {
    background: #0b0c10;         /* <- default to dark everywhere */
    color: #e6edf3;
}

/* Scroll area needs both the abstract and viewport styled */
QAbstractScrollArea,
QAbstractScrollArea::viewport,
QScrollArea,
QScrollArea QWidget {
    background: #0b0c10;
}

/* Right panel explicit (in case you name the container) */
#RightPanel {
    background: #0b0c10;
}

/* Card-like groups */
QGroupBox {
    margin-top: 14px;
    border: 1px solid #2b2f36;
    border-radius: 10px;
    padding: 12px;
    background-color: #1c1f26;   /* lighter card on dark surface */
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    background: transparent;
}

/* Labels/inputs */
QLabel, QCheckBox, QSpinBox, QDoubleSpinBox {
    color: #e6edf3;
    font-size: 13px;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #2b2f36;
    height: 6px;
    border-radius: 3px;
    background: #2b2f36;
}
QSlider::handle:horizontal {
    background: #1f6feb;
    border: none;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #1f6feb;
    border-radius: 3px;
}

/* Buttons */
QPushButton {
    background: #1f6feb;
    color: white;
    border: 0;
    padding: 8px 12px;
    border-radius: 8px;
}
QPushButton:disabled {
    background: #334155;
    color: #9aa4ad;
}

/* Progress bar */
QProgressBar {
    background: #111318;
    border: 1px solid #2b2f36;
    border-radius: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: #1f6feb;
    border-radius: 8px;
}

/* primary CTA button */
QPushButton#btn_convert {
    background-color: #1f6feb;
    color: #ffffff;
    border: 1px solid #2b2f36;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
}
QPushButton#btn_convert:disabled {
    background-color: #334155;
    color: #9aa4ad;
}
"""

# --------------- clickable label for upload ----------------------------
class ClickableLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)

# ── main window ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("String Art — GUI")
        self.resize(1500, 900)
        self.setStyleSheet(APP_STYLES)

        self.img_bgr: Optional[np.ndarray] = None
        self.current_path: List[Segment] = []
        self.current_pins: Optional[np.ndarray] = None
        self.current_work_size: int = 0

        self._thread: Optional[QThread] = None
        self._worker: Optional[ConvertWorker] = None

        self._build_ui()
        self._build_menu()

        self.setAcceptDrops(True)

    # ── UI layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        main = QHBoxLayout(root)
        main.setContentsMargins(14, 12, 14, 12)
        main.setSpacing(14)

        # Left: image preview
        self.image_label = ClickableLabel("Drop an image here or use File → Open…")
        self.image_label.setObjectName("HintLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(680, 480)
        self.image_label.setStyleSheet("border:1px dashed #888; border-radius:12px; padding:16px;")
        self.image_label.setCursor(Qt.PointingHandCursor)   # nice UX hint
        self.image_label.clicked.connect(self.open_image)

        # Right: controls (scrollable)
        right_panel = QWidget()
        right_panel.setObjectName("RightPanel")                # <- name it so CSS can target it
        right_panel.setAttribute(Qt.WA_StyledBackground, True) # <- ensure bg is painted
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        title = QLabel("Controls")
        title.setObjectName("TitleLabel")
        right_layout.addWidget(title)

        self.combo_preset = QComboBox()
        self.combo_preset.setObjectName("comboPreset")
        self.combo_preset.addItem("— None —")  # will be populated after widgets exist
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        row.addWidget(self.combo_preset, 1)
        right_layout.addLayout(row)

        right_layout.addWidget(self._group_source())
        right_layout.addWidget(self._group_solver())
        right_layout.addWidget(self._group_preview())

        # Run row
        run_row = QHBoxLayout()
        self.btn_convert = QPushButton("Start Conversion")
        self.btn_convert.setObjectName("btn_convert")
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_convert.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        run_row.addWidget(self.btn_convert, 0)
        run_row.addWidget(self.progress, 1)
        right_layout.addLayout(run_row)
        right_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        # not strictly required if using the stylesheet above, but safe:
        scroll.viewport().setAttribute(Qt.WA_StyledBackground, True)
        scroll.setWidget(right_panel)

        main.addWidget(self.image_label, 2)
        main.addWidget(scroll, 1)
        self.setCentralWidget(root)

        # connect presets
        self._build_wmap()
        self._load_presets_json()           # load ranges + defaults + presets
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)

    def _group_source(self) -> QGroupBox:
        g = QGroupBox("Image / Preprocessing")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignRight)

        # work_size
        self.spin_work = QSpinBox(); self.spin_work.setRange(128, 2048); self.spin_work.setValue(500)
        f.addRow("Work size (px):", self.spin_work)

        # CLAHE
        self.chk_clahe = QCheckBox("CLAHE (local contrast)")
        f.addRow("", self.chk_clahe)

        # Contrast stretch
        self.chk_contrast = QCheckBox("Contrast stretch (percentiles)")
        self.dsp_low = QDoubleSpinBox(); self.dsp_low.setRange(0, 50); self.dsp_low.setDecimals(1); self.dsp_low.setValue(0.0)
        self.dsp_high = QDoubleSpinBox(); self.dsp_high.setRange(50, 100); self.dsp_high.setDecimals(1); self.dsp_high.setValue(80.0)
        f.addRow(self.chk_contrast)
        f.addRow("Low %:", self.dsp_low)
        f.addRow("High %:", self.dsp_high)

        # Edges
        self.chk_edges = QCheckBox("Blend Canny edges")
        self.dsp_edge_weight = QDoubleSpinBox(); self.dsp_edge_weight.setRange(0, 1); self.dsp_edge_weight.setSingleStep(0.05); self.dsp_edge_weight.setValue(0.35)
        f.addRow(self.chk_edges)
        f.addRow("Edge weight:", self.dsp_edge_weight)

        # Background dim via rembg
        self.chk_rembg = QCheckBox("Darken background (rembg)")
        self.dsp_rembg_dim = QDoubleSpinBox(); self.dsp_rembg_dim.setRange(0, 1); self.dsp_rembg_dim.setSingleStep(0.05); self.dsp_rembg_dim.setValue(0.6)
        self.spin_rembg_feather = QSpinBox(); self.spin_rembg_feather.setRange(0, 64); self.spin_rembg_feather.setValue(8)
        self.spin_rembg_erode = QSpinBox(); self.spin_rembg_erode.setRange(0, 8); self.spin_rembg_erode.setValue(1)

        f.addRow(self.chk_rembg)
        f.addRow("Dim (0–1):", self.dsp_rembg_dim)
        f.addRow("Feather σ (px):", self.spin_rembg_feather)
        f.addRow("Erode FG (px):", self.spin_rembg_erode)

        return g

    def _group_solver(self) -> QGroupBox:
        g = QGroupBox("Solver")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignRight)

        self.spin_pins = QSpinBox(); self.spin_pins.setRange(12, 2048); self.spin_pins.setValue(300)
        self.spin_steps = QSpinBox(); self.spin_steps.setRange(1, 20000); self.spin_steps.setValue(4000)
        self.spin_min_dist = QSpinBox(); self.spin_min_dist.setRange(0, 512); self.spin_min_dist.setValue(30)
        self.dsp_line_weight = QDoubleSpinBox(); self.dsp_line_weight.setRange(0.1, 64.0); self.dsp_line_weight.setDecimals(3); self.dsp_line_weight.setValue(8.0)
        self.spin_lastn = QSpinBox(); self.spin_lastn.setRange(0, 256); self.spin_lastn.setValue(20)

        f.addRow("Pins:", self.spin_pins)
        f.addRow("Max lines:", self.spin_steps)
        f.addRow("Min distance:", self.spin_min_dist)
        f.addRow("Line weight:", self.dsp_line_weight)
        f.addRow("Cooldown last-N:", self.spin_lastn)

        return g

    def _group_preview(self) -> QGroupBox:
        g = QGroupBox("Preview")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignRight)

        self.dsp_alpha = QDoubleSpinBox(); self.dsp_alpha.setRange(0.005, 0.5); self.dsp_alpha.setDecimals(3); self.dsp_alpha.setValue(0.1)
        self.dsp_gamma = QDoubleSpinBox(); self.dsp_gamma.setRange(0.5, 3.0); self.dsp_gamma.setSingleStep(0.05); self.dsp_gamma.setValue(1.20)
        self.spin_thick = QSpinBox(); self.spin_thick.setRange(1, 5); self.spin_thick.setValue(1)

        f.addRow("Darken per line:", self.dsp_alpha)
        f.addRow("Gamma:", self.dsp_gamma)
        f.addRow("Line thickness:", self.spin_thick)

        # buttons
        row = QHBoxLayout()
        self.btn_save_preview = QPushButton("Save Preview…")
        self.btn_save_preview.clicked.connect(self.save_preview)
        self.btn_save_preview.setEnabled(False)
        self.btn_export_path = QPushButton("Export Path (CSV)…")
        self.btn_export_path.clicked.connect(self.export_path)
        self.btn_export_path.setEnabled(False)
        row.addWidget(self.btn_save_preview)
        row.addWidget(self.btn_export_path)
        f.addRow(row)
        return g

    # ── menu ─────────────────────────────────────────────────────────────────
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        act_open = QAction("Open Image…", self)
        act_open.triggered.connect(self.open_image)
        m.addAction(act_open)

        m.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        m.addAction(act_quit)

    # ── drag & drop ──────────────────────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            self.load_image(url.toLocalFile())
            break

    # ── file ops ─────────────────────────────────────────────────────────────
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            warn(self, "Open image", "Could not read the image.")
            return
        self.img_bgr = bgr
        rgb = bgr_to_rgb_img(bgr)
        self.image_label.setPixmap(to_qpixmap_from_rgb(rgb, self.image_label.size()))
        self.btn_convert.setEnabled(True)
        self.btn_save_preview.setEnabled(False)
        self.btn_export_path.setEnabled(False)
        self.current_path = []
        self.progress.setValue(0)

    # ── run conversion ───────────────────────────────────────────────────────
    def gather_params(self) -> dict:
        return dict(
            work_size=self.spin_work.value(),
            pins=self.spin_pins.value(),
            steps=self.spin_steps.value(),
            min_distance=self.spin_min_dist.value(),
            line_weight=float(self.dsp_line_weight.value()),
            last_n=self.spin_lastn.value(),
            # preprocessing
            pp_clahe=self.chk_clahe.isChecked(),
            pp_contrast=self.chk_contrast.isChecked(),
            pp_c_low=float(self.dsp_low.value()),
            pp_c_high=float(self.dsp_high.value()),
            pp_edges=self.chk_edges.isChecked(),
            pp_edge_weight=float(self.dsp_edge_weight.value()),
            pp_edge_low=-1,           # keep auto
            pp_edge_high=-1,          # keep auto
            pp_edge_auto_sigma=0.33,
            pp_rembg=self.chk_rembg.isChecked(),
            pp_rembg_dim=float(self.dsp_rembg_dim.value()),
            pp_rembg_feather=self.spin_rembg_feather.value(),
            pp_rembg_erode=self.spin_rembg_erode.value(),
            # preview 
            render_alpha=float(self.dsp_alpha.value()),
            render_gamma=float(self.dsp_gamma.value()),
            line_thickness=self.spin_thick.value(),
        )

    def start_conversion(self):
        if self.img_bgr is None:
            info(self, "No image", "Load an image first.")
            return

        params = self.gather_params()

        # stop existing thread if any
        if self._thread:
            self._thread.quit()
            self._thread.wait()

        self._thread = QThread()
        self._worker = ConvertWorker(self.img_bgr, params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(self.on_finished)
        self._worker.errored.connect(self.on_errored)
        self._worker.finished.connect(self._thread.quit)
        self._worker.errored.connect(self._thread.quit)

        self.btn_convert.setEnabled(False)
        self._thread.start()

    def on_finished(self, path: List[Segment], err: np.ndarray, target: np.ndarray, pins: np.ndarray):
        self.current_path = path
        self.current_pins = pins
        self.current_work_size = target.shape[0] if target.ndim == 2 else int(math.sqrt(target.size))

        self.btn_convert.setEnabled(True)
        self.btn_save_preview.setEnabled(bool(path))
        self.btn_export_path.setEnabled(bool(path))

        if not path:
            info(self, "Conversion", "No path produced (try fewer pins or more lines).")
            return

        # render preview
        params = self.gather_params()
        preview_u8 = render_path(
            work_size=self.current_work_size,
            pins=pins,
            path=path,
            alpha_per_line=params["render_alpha"],
            gamma=params["render_gamma"],
            thickness=params["line_thickness"],
        )
        rgb = np.dstack([preview_u8]*3)  # grayscale → RGB for display
        self.image_label.setPixmap(to_qpixmap_from_rgb(rgb, self.image_label.size()))
        info(self, "Done", f"Generated {len(path)} segments.")

    def on_errored(self, msg: str):
        self.btn_convert.setEnabled(True)
        error(self, "Error during conversion", msg)

    # ── exports ──────────────────────────────────────────────────────────────
    def save_preview(self):
        if not self.current_path or self.current_pins is None:
            info(self, "No preview", "Run a conversion first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save preview", "preview.png", "PNG (*.png)")
        if not path:
            return
        params = self.gather_params()
        preview = render_path(
            work_size=self.current_work_size,
            pins=self.current_pins,
            path=self.current_path,
            alpha_per_line=params["render_alpha"],
            gamma=params["render_gamma"],
            thickness=params["line_thickness"],
        )
        cv2.imwrite(path, preview)
        info(self, "Saved", f"Preview saved to:\n{path}")

    def export_path(self):
        if not self.current_path:
            info(self, "Nothing to export", "Run a conversion first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save coordinates", "path.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["from_pin", "to_pin"]); w.writerows(self.current_path)
            info(self, "Saved", f"Coordinates saved to:\n{path}")
        except Exception as e:
            error(self, "Save failed", str(e))

    # ── import jsons ──────────────────────────────────────────────────────────────
    def _build_wmap(self):
        """Map parameter keys <-> widgets (must match your utils/json keys)."""
        self.wmap = {
            # core / solver
            "work_size":         self.spin_work,
            "pins":              self.spin_pins,
            "steps":             self.spin_steps,
            "min_distance":      self.spin_min_dist,
            "line_weight":       self.dsp_line_weight,
            "last_n":            self.spin_lastn,

            # preview
            "render_alpha":      self.dsp_alpha,
            "render_gamma":      self.dsp_gamma,
            "line_thickness":    self.spin_thick,

            # preprocessing
            "pp_clahe":          self.chk_clahe,
            "pp_contrast":       self.chk_contrast,
            "pp_c_low":          self.dsp_low,
            "pp_c_high":         self.dsp_high,
            "pp_edges":          self.chk_edges,
            "pp_edge_weight":    self.dsp_edge_weight,
            # you currently keep auto thresholds:
            # "pp_edge_low":     <no widget>,
            # "pp_edge_high":    <no widget>,
            # "pp_edge_auto_sigma": <constant in gather_params>,

            "pp_rembg":          self.chk_rembg,
            "pp_rembg_dim":      self.dsp_rembg_dim,
            "pp_rembg_feather":  self.spin_rembg_feather,
            "pp_rembg_erode":    self.spin_rembg_erode,
        }

    def _load_presets_json(self):
        """Read settings.json, populate ranges, defaults, and preset list."""
        # 1) load file
        cfg_path = os.path.join(os.path.dirname(__file__) + "\\StringArtConverter", "settings.json")
        self._cfg = load_presets_json(cfg_path)  # {ranges, defaults, presets}

        # 2) apply ranges to widgets
        set_widget_ranges(self._cfg.get("ranges", {}), self.wmap)

        # 3) apply defaults
        defaults = clamp_to_ranges(self._cfg.get("defaults", {}), self._cfg.get("ranges", {}))
        apply_to_widgets(defaults, self.wmap)

        # 4) fill preset combo
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        self.combo_preset.addItem("— None —")
        for p in self._cfg.get("presets", []):
            self.combo_preset.addItem(p.get("name", "Untitled"))
        self.combo_preset.blockSignals(False)

    def _on_preset_changed(self, idx: int):
        """Apply a named preset (defaults merged with preset params)."""
        if idx <= 0:
            # back to defaults
            defaults = clamp_to_ranges(self._cfg.get("defaults", {}), self._cfg.get("ranges", {}))
            apply_to_widgets(defaults, self.wmap)
            return
        preset = self._cfg["presets"][idx - 1]
        params = dict(self._cfg.get("defaults", {}))
        params.update(preset.get("params", {}))
        params = clamp_to_ranges(params, self._cfg.get("ranges", {}))
        apply_to_widgets(params, self.wmap)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("String Art (Go-style) — GUI")

    # optional window icon if you have one:
    # app.setWindowIcon(QIcon("icon.png"))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()