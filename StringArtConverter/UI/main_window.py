# region ----------- imports -----------
from __future__ import annotations
from typing import List, Optional
import math
import cv2
import numpy as np
import os
from itertools import product

# -------- UI ----------
from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize
from PySide6.QtGui import QAction, QPixmap, QImage, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QProgressBar, QMessageBox, QScrollArea, QFrame, QComboBox, QHBoxLayout
)
from StringArtConverter.UI.sliders import IntSlider, FloatSlider
from StringArtConverter.UI.ui_utils import ClickableLabel, apply_to_widgets, collect_from_widgets, set_widget_ranges

# -------- solver imports --------
from StringArtConverter.preprocessing import build_brightness_for_go_solver
from StringArtConverter.utils import save_path_txt, load_presets_json, clamp_to_ranges, Segment
from StringArtConverter.previewer import render_path
from StringArtConverter.solver import solve_string_art_go

# -------- APP STYLES --------
from StringArtConverter.UI.app_styles import APP_STYLES
# endregion

# region ----------- helpers -----------
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

# endregion

# region ----------- worker -----------
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
# endregion

# region ----------- param search worker -----------
class BatchSearchWorker(QObject):
    progress = Signal(int)   # overall %
    finished = Signal(str)   # out_dir
    errored = Signal(str)

    def __init__(self, img_bgr: np.ndarray, base_params: dict, grid: dict, out_dir: str):
        super().__init__()
        self.img_bgr = img_bgr
        self.base = base_params
        self.grid = grid
        self.out_dir = out_dir

    def _variants(self):
        keys = list(self.grid.keys())
        vals = [self.grid[k] for k in keys]
        for combo in product(*vals):
            p = dict(self.base)
            for k, v in zip(keys, combo):
                p[k] = v
            yield p

    def run(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            lines = []
            total = 0
            for _ in self._variants():
                total += 1
            if total == 0:
                self.errored.emit("Grid is empty.")
                return

            # regenerate iterator (consumed)
            idx = 0
            for params in self._variants():
                idx += 1

                # ---- preprocessing
                src_u8 = build_brightness_for_go_solver(
                    img_bgr=self.img_bgr,
                    work_size=params["work_size"],
                    use_clahe=params["pp_clahe"],
                    use_contrast=params["pp_contrast"],
                    p_low=params["pp_c_low"],
                    p_high=params["pp_c_high"],
                    use_edges=params["pp_edges"],
                    edge_weight=params["pp_edge_weight"],
                    edge_low=-1, edge_high=-1, edge_auto_sigma=0.33,
                    use_rembg=params["pp_rembg"],
                    rembg_dim=params["pp_rembg_dim"],
                    rembg_feather=params["pp_rembg_feather"],
                    rembg_erode=params["pp_rembg_erode"],
                    pp_gamma=params.get("pp_gamma", 1.0),
                    pp_clip_high=params.get("pp_clip_high", 100.0),
                )

                # ---- solve
                path, err, target, pins = solve_string_art_go(
                    source_brightness_u8=src_u8,
                    n_pins=params["pins"],
                    max_lines=params["steps"],
                    min_distance=params["min_distance"],
                    line_weight=params["line_weight"],
                    last_n=params["last_n"],
                    work_size=params["work_size"],
                    progress_cb=None,  # per-run progress omitted; we show overall only
                )

                # ---- preview
                preview = render_path(
                    work_size=params["work_size"],
                    pins=pins,
                    path=path,
                    alpha_per_line=params["render_alpha"],
                    gamma=params["render_gamma"],
                    thickness=params["line_thickness"],
                )

                img_name = f"{idx}.png"
                cv2.imwrite(os.path.join(self.out_dir, img_name), preview)

                # simple score: lower mean error is better (post-solve)
                score = float(err.mean()) if err is not None else float("nan")

                # record params (only the ones that vary or matter visually)
                rec_keys = [
                    "work_size","pins","steps","min_distance","line_weight","last_n",
                    "pp_clahe","pp_contrast","pp_c_low","pp_c_high",
                    "pp_edges","pp_edge_weight",
                    "pp_rembg","pp_rembg_dim","pp_rembg_feather","pp_rembg_erode",
                    "pp_gamma","pp_clip_high",
                    "render_alpha","render_gamma","line_thickness"
                ]
                snapshot = ", ".join(f"{k}={params.get(k)}" for k in rec_keys)
                lines.append(f"{idx}: {img_name} | lines={len(path)} | score={score:.6f} | {snapshot}")

                self.progress.emit(int(100 * idx / total))

            with open(os.path.join(self.out_dir, "params.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            self.finished.emit(self.out_dir)

        except Exception as e:
            self.errored.emit(str(e))
# endregion

# region ----------- Main window -----------
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

        # Run row
        run_row = QHBoxLayout()
        self.btn_convert = QPushButton("Start Conversion")
        self.btn_convert.setObjectName("btn_convert")
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_convert.setEnabled(False)

        # params button
        self.btn_batch = QPushButton("Batch Preset Search…")
        self.btn_batch.clicked.connect(self.start_batch_search)
        self.btn_batch.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        run_row.addWidget(self.btn_convert, 0)
        run_row.addWidget(self.btn_batch, 0)
        run_row.addWidget(self.progress, 1)
        right_layout.addLayout(run_row)
        right_layout.addStretch(1)

        # controls
        title = QLabel("Controls")
        title.setObjectName("TitleLabel")
        right_layout.addWidget(title)

        # preview
        right_layout.addWidget(self._group_preview())

        # presets
        self.combo_preset = QComboBox()
        self.combo_preset.setObjectName("comboPreset")
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        row.addWidget(self.combo_preset, 1)
        right_layout.addLayout(row)
        # solver settings
        right_layout.addWidget(self._group_solver())
        # preprocessing settings
        right_layout.addWidget(self._group_source())
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        # not strictly required if using stylesheet, but safe:
        scroll.viewport().setAttribute(Qt.WA_StyledBackground, True)
        scroll.setWidget(right_panel)

        main.addWidget(self.image_label, 2)
        main.addWidget(scroll, 1)
        self.setCentralWidget(root)

        # connect presets
        self._build_wmap()
        self._load_presets_json()           # load ranges + defaults + presets
        self.combo_preset.currentIndexChanged[int].connect(self._on_preset_changed)

    def _group_source(self) -> QGroupBox:
        g = QGroupBox("Image preprocessing options")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignRight)

        # Work size (int slider)
        self.sld_work = IntSlider(128, 2048, 500, suffix=" px", tick=128)
        f.addRow("Work size:", self.sld_work)

        # CLAHE
        self.chk_clahe = QCheckBox("CLAHE")
        f.addRow(self.chk_clahe)

        # Contrast stretch (float sliders)
        self.chk_contrast = QCheckBox("Contrast stretch")
        self.sld_low = FloatSlider(0.0, 50.0, 0.0, step=0.5, suffix=" %")
        self.sld_high = FloatSlider(50.0, 100.0, 80.0, step=0.5, suffix=" %")
        f.addRow(self.chk_contrast)
        f.addRow("Low percentile:", self.sld_low)
        f.addRow("High percentile:", self.sld_high)

        # Edges
        self.chk_edges = QCheckBox("Blend edges")
        self.sld_edge_weight = FloatSlider(0.0, 1.0, 0.35, step=0.01)
        f.addRow(self.chk_edges)
        f.addRow("Edge weight:", self.sld_edge_weight)

        # Background dim via rembg
        self.chk_rembg = QCheckBox("Darken background")
        self.sld_rembg_dim = FloatSlider(0.0, 1.0, 0.6, step=0.05)
        self.sld_rembg_feather = IntSlider(0, 64, 8, suffix=" px", tick=4)
        self.sld_rembg_erode = IntSlider(0, 8, 1, suffix=" px")
        f.addRow(self.chk_rembg)
        f.addRow("Dim factor:", self.sld_rembg_dim)
        f.addRow("Feather σ:", self.sld_rembg_feather)
        f.addRow("Erode FG:", self.sld_rembg_erode)

        # Check for ranges of sliders
        self._wire_percentile_guards()

        return g

    def _group_solver(self) -> QGroupBox:
        g = QGroupBox("General options")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignRight)

        self.sld_pins = IntSlider(12, 2048, 300)
        self.sld_steps = IntSlider(1, 20000, 4000, tick=1000)
        self.sld_min_dist = IntSlider(0, 512, 30, suffix=" pins", tick=16)
        # line_weight can work well on 0.1..16 for most photos; adjust if you like
        self.sld_line_weight = FloatSlider(0.1, 16.0, 8.0, step=0.01)
        self.sld_lastn = IntSlider(0, 256, 20)

        f.addRow("Pins:", self.sld_pins)
        f.addRow("Max lines:", self.sld_steps)
        f.addRow("Min distance:", self.sld_min_dist)
        f.addRow("Line weight:", self.sld_line_weight)
        f.addRow("Cooldown last-N:", self.sld_lastn)
        return g

    def _group_preview(self) -> QGroupBox:
        g = QGroupBox("Preview options")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignRight)

        self.sld_alpha = FloatSlider(0.005, 0.5, 0.10, step=0.005)
        self.sld_gamma = FloatSlider(0.5, 3.0, 1.20, step=0.05)
        self.sld_thick = IntSlider(1, 5, 1)

        f.addRow("Darken per string:", self.sld_alpha)
        f.addRow("Gamma:", self.sld_gamma)
        f.addRow("Line thickness:", self.sld_thick)

        # buttons row stays the same
        row = QHBoxLayout()
        self.btn_save_preview = QPushButton("Save Preview…")
        self.btn_save_preview.clicked.connect(self.save_preview)
        self.btn_save_preview.setEnabled(False)
        self.btn_export_path = QPushButton("Export Path…")
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
    
    def _wire_percentile_guards(self):
        def clamp_low(v):
            if v > self.sld_high.value():
                self.sld_low.setValue(self.sld_high.value())
        def clamp_high(v):
            if v < self.sld_low.value():
                self.sld_high.setValue(self.sld_low.value())

        self.sld_low.valueChanged.connect(clamp_low)
        self.sld_high.valueChanged.connect(clamp_high)

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
        self.btn_batch.setEnabled(True)
        self.btn_save_preview.setEnabled(False)
        self.btn_export_path.setEnabled(False)
        self.current_path = []
        self.progress.setValue(0)

    # ── run conversion ───────────────────────────────────────────────────────
    def gather_params(self) -> dict:
        return dict(
            work_size=self.sld_work.value(),
            pins=self.sld_pins.value(),
            steps=self.sld_steps.value(),
            min_distance=self.sld_min_dist.value(),
            line_weight=float(self.sld_line_weight.value()),
            last_n=self.sld_lastn.value(),
            # preprocessing
            pp_clahe=self.chk_clahe.isChecked(),
            pp_contrast=self.chk_contrast.isChecked(),
            pp_c_low=float(self.sld_low.value()),
            pp_c_high=float(self.sld_high.value()),
            pp_edges=self.chk_edges.isChecked(),
            pp_edge_weight=float(self.sld_edge_weight.value()),
            pp_edge_low=-1,
            pp_edge_high=-1,
            pp_edge_auto_sigma=0.33,
            pp_rembg=self.chk_rembg.isChecked(),
            pp_rembg_dim=float(self.sld_rembg_dim.value()),
            pp_rembg_feather=self.sld_rembg_feather.value(),
            pp_rembg_erode=self.sld_rembg_erode.value(),
            # preview
            render_alpha=float(self.sld_alpha.value()),
            render_gamma=float(self.sld_gamma.value()),
            line_thickness=self.sld_thick.value(),
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
            "work_size":         self.sld_work,
            "pins":              self.sld_pins,
            "steps":             self.sld_steps,
            "min_distance":      self.sld_min_dist,
            "line_weight":       self.sld_line_weight,
            "last_n":            self.sld_lastn,

            # preview
            "render_alpha":      self.sld_alpha,
            "render_gamma":      self.sld_gamma,
            "line_thickness":    self.sld_thick,

            # preprocessing
            "pp_clahe":          self.chk_clahe,
            "pp_contrast":       self.chk_contrast,
            "pp_c_low":          self.sld_low,
            "pp_c_high":         self.sld_high,
            "pp_edges":          self.chk_edges,
            "pp_edge_weight":    self.sld_edge_weight,
            # you currently keep auto thresholds:
            # "pp_edge_low":     <no widget>,
            # "pp_edge_high":    <no widget>,
            # "pp_edge_auto_sigma": <constant in gather_params>,

            "pp_rembg":          self.chk_rembg,
            "pp_rembg_dim":      self.sld_rembg_dim,
            "pp_rembg_feather":  self.sld_rembg_feather,
            "pp_rembg_erode":    self.sld_rembg_erode,
        }

    def _load_presets_json(self):
        """Read settings.json, populate ranges, defaults, and preset list."""
        # load file
        cfg_path = os.path.join(os.path.dirname(__file__), "settings.json")
        self._cfg = load_presets_json(cfg_path)  # {ranges, defaults, presets}

        # apply ranges to widgets
        set_widget_ranges(self._cfg.get("ranges", {}), self.wmap)

        # apply defaults
        defaults = clamp_to_ranges(self._cfg.get("defaults", {}), self._cfg.get("ranges", {}))
        apply_to_widgets(defaults, self.wmap)

        # fill preset combo
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        self.combo_preset.addItem("Defaults")
        for p in self._cfg.get("presets", []):
            self.combo_preset.addItem(p.get("name", "Untitled"))
        self.combo_preset.blockSignals(False)

        self.combo_preset.setCurrentIndex(0)
        self._on_preset_changed(0)

    def _on_preset_changed(self, idx: int):
        if not hasattr(self, "_cfg"):  # safety
            return
        ranges = self._cfg.get("ranges", {})
        if idx == 0:
            params = clamp_to_ranges(self._cfg.get("defaults", {}), ranges)
        else:
            preset = self._cfg["presets"][idx - 1]
            params = dict(self._cfg.get("defaults", {}))
            params.update(preset.get("params", {}))
            params = clamp_to_ranges(params, ranges)
        apply_to_widgets(params, self.wmap)

    # --------------- Batch prams debug/get -----------------------
    def start_batch_search(self):
        if self.img_bgr is None:
            info(self, "No image", "Load an image first.")
            return

        # choose output folder
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder for batch results")
        if not out_dir:
            return

        base = self.gather_params()

        # --------- EDIT THIS GRID as you like (kept modest to avoid explosion)
        grid = {
            # preprocessing toggles/weights
            "pp_edges":        [True],
            "pp_edge_weight":  [0.25],

            "pp_clahe":        [False],
            "pp_contrast":     [False, True],
            "pp_c_low":        [2.0, 5.0, 10.0],
            "pp_c_high":       [100.0, 95.0, 90.0],

            "pp_rembg":        [True],
            "pp_rembg_dim":    [0.3],
            "pp_rembg_feather":[base["pp_rembg_feather"]],
            "pp_rembg_erode":  [base["pp_rembg_erode"]],

            # tonal shaping
            "pp_gamma":        [0.65],
            "pp_clip_high":    [95.0],         

            # solver minimal variation (or stick to base to keep runtime sane)
            "line_weight":     [8],
            "min_distance":    [base["min_distance"]],
            # keep pins/steps/last_n from base:
            "pins":            [base["pins"]],
            "steps":           [base["steps"]],
            "last_n":          [base["last_n"]],
            "work_size":       [base["work_size"]],

            # preview kept constant so visual comparison is fair:
            "render_alpha":    [base["render_alpha"]],
            "render_gamma":    [base["render_gamma"]],
            "line_thickness":  [base["line_thickness"]],
        }

        def _prune_grid(base: dict, grid: dict) -> dict:
            """Return a copy of grid with dependent knobs collapsed when their toggle is False."""
            g = {k: list(v) for k, v in grid.items()}  # shallow copy of lists

            # If pp_edges includes False, collapse pp_edge_weight for the False branch.
            # Easiest practical way: if pp_edges is [False] only, just keep one value for weight.
            if "pp_edges" in g and g["pp_edges"] == [False]:
                # keep current UI value (or first provided) to avoid extra runs
                g["pp_edge_weight"] = [base.get("pp_edge_weight", g.get("pp_edge_weight", [0.35])[0])]

            # If pp_rembg is [False], collapse its extras
            if "pp_rembg" in g and g["pp_rembg"] == [False]:
                g["pp_rembg_dim"] = [base.get("pp_rembg_dim", g.get("pp_rembg_dim", [0.0])[0])]
                g["pp_rembg_feather"] = [base.get("pp_rembg_feather", g.get("pp_rembg_feather", [8])[0])]
                g["pp_rembg_erode"] = [base.get("pp_rembg_erode", g.get("pp_rembg_erode", [1])[0])]

            return g

        grid = _prune_grid(base, grid)


        # Optional: show how many runs
        def _count_runs(g):
            c = 1
            for v in g.values():
                c *= max(1, len(v))
            return c

        runs = _count_runs(grid)
        confirm = QMessageBox.question(
            self, "Batch Preset Search",
            f"This will render {runs} variants.\nProceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if confirm != QMessageBox.Yes:
            return

        # stop existing worker if any
        if self._thread:
            self._thread.quit()
            self._thread.wait()

        self._thread = QThread()
        self._batch_worker = BatchSearchWorker(self.img_bgr, base, grid, out_dir)
        self._batch_worker.moveToThread(self._thread)

        # connect
        self._thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self.progress.setValue)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.errored.connect(self.on_errored)
        self._batch_worker.finished.connect(self._thread.quit)
        self._batch_worker.errored.connect(self._thread.quit)

        # UI lock
        self.btn_convert.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.progress.setValue(0)
        self._thread.start()

    def _on_batch_finished(self, out_dir: str):
        self.btn_convert.setEnabled(True)
        self.btn_batch.setEnabled(True)
        info(self, "Batch done", f"Saved previews and params.txt to:\n{out_dir}")

    
# endregion