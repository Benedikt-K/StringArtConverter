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
from PySide6.QtGui import QAction, QPixmap, QImage
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QCheckBox,QGroupBox, QProgressBar, 
    QMessageBox, QScrollArea, QFrame, QComboBox, QHBoxLayout, QSpinBox, QFormLayout
)
from StringArtConverter.UI.sliders import IntSlider, FloatSlider
from StringArtConverter.UI.ui_utils import ClickableLabel, CardGroup, apply_to_widgets, set_widget_ranges, add_card_shadow

# -------- solver imports --------
from StringArtConverter.preprocessing import build_brightness_for_solver
from StringArtConverter.utils import load_presets_json, clamp_to_ranges, read_path_csv, save_session_json, load_session_json, Segment
from StringArtConverter.previewer import render_path
from StringArtConverter.solver import solve_string_art_go, pin_positions_circle

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

# region ----------- convert worker -----------
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
            src_u8 = build_brightness_for_solver(
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
    progress = Signal(int)
    finished = Signal(str)
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

            idx = 0
            for params in self._variants():
                idx += 1

                src_u8 = build_brightness_for_solver(
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

                path, err, target, pins = solve_string_art_go(
                    source_brightness_u8=src_u8,
                    n_pins=params["pins"],
                    max_lines=params["steps"],
                    min_distance=params["min_distance"],
                    line_weight=params["line_weight"],
                    last_n=params["last_n"],
                    work_size=params["work_size"],
                    progress_cb=None,
                )

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

                score = float(err.mean()) if err is not None else float("nan")

                # record only relevant params
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
        self.setWindowTitle("String Art Converter")
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

        self.guided_path: list[tuple[int, int]] = []
        self.guided_pins: Optional[np.ndarray] = None
        self.guided_work_size: int = 0
        self.guided_index: int = -1

        self.setAcceptDrops(True)

    # ----------- UI layout -----------
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
        self.image_label.setCursor(Qt.PointingHandCursor)
        self.image_label.clicked.connect(self.open_image)

        # Right: controls (scrollable)
        right_panel = QWidget()
        right_panel.setObjectName("RightPanel")
        right_panel.setAttribute(Qt.WA_StyledBackground, True)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        # Run row
        run_row = QHBoxLayout()
        self.btn_convert = QPushButton("Start Conversion")
        self.btn_convert.setObjectName("btn_convert")
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_convert.setEnabled(False)

        # params search button
        self.btn_batch = QPushButton("Batch Preset Search…")
        self.btn_batch.clicked.connect(self.start_batch_search)
        self.btn_batch.setEnabled(False)

        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        run_row.addWidget(self.btn_convert, 0)
        run_row.addWidget(self.btn_batch, 0)
        run_row.addWidget(self.progress, 1)
        right_layout.addLayout(run_row)
        right_layout.addStretch(1)
        right_layout.setAlignment(Qt.AlignTop)

        # preview
        self.group_preview = self._group_preview()
        right_layout.addWidget(self.group_preview)

        # presets
        self.combo_preset = QComboBox()
        self.combo_preset.setObjectName("comboPreset")
        row = QHBoxLayout()
        row.addWidget(QLabel("Presets:"))
        row.addWidget(self.combo_preset, 1)
        right_layout.addLayout(row)

        # solver settings
        self.group_solver = self._group_solver()
        right_layout.addWidget(self.group_solver)

        # preprocessing settings
        self.group_source  = self._group_source()
        right_layout.addWidget(self.group_source)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.viewport().setAttribute(Qt.WA_StyledBackground, True)
        scroll.setWidget(right_panel)

        # guided
        self.group_guided = self._group_guided()
        right_layout.addWidget(self.group_guided)

        main.addWidget(self.image_label, 2)
        main.addWidget(scroll, 1)
        self.setCentralWidget(root)

        # add card shadows
        add_card_shadow(self.group_preview)
        add_card_shadow(self.group_solver)
        add_card_shadow(self.group_source)
        add_card_shadow(self.combo_preset)
        add_card_shadow(self.group_guided)

        # connect presets
        self._build_wmap()
        self._load_presets_json()
        self.combo_preset.currentIndexChanged[int].connect(self._on_preset_changed)
        self._set_guided_enabled(False)

    def _group_source(self) -> QGroupBox:
        help_html = (
            "<b>Preprocessing</b><br>"
            "<u>Work size</u>: resizing the image (higher = better quality and slower).<br>"
            "<u>CLAHE</u>: local contrast equalization (can add noise).<br>"
            "<u>Contrast stretch</u>: remap dark/bright percentiles ('compress' grayscale values).<br>"
            "<u>Blend edges</u>: mix edges into the target (higher = more contour bias).<br>"
            "<u>Darken background (rembg)</u>: AI background detection mask to dim background;"
            "<u>Feather</u>: softens the mask edges.<br>"
            "<u>Erode</u>: shriks the mask."
        )
        card = CardGroup("Image preprocessing options", help_html, self)
        f = card.form

        # Work size
        self.sld_work = IntSlider(128, 2048, 500, suffix=" px", tick=128)
        f.addRow("Work size:", self.sld_work)

        # CLAHE
        self.chk_clahe = QCheckBox("CLAHE")
        f.addRow(self.chk_clahe)

        # Contrast stretch
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

        # Background dim
        self.chk_rembg = QCheckBox("Darken background")
        self.sld_rembg_dim = FloatSlider(0.0, 1.0, 0.6, step=0.05)
        self.sld_rembg_feather = IntSlider(0, 64, 8, suffix=" px", tick=4)
        self.sld_rembg_erode = IntSlider(0, 8, 1, suffix=" px")
        f.addRow(self.chk_rembg)
        f.addRow("Dim factor:", self.sld_rembg_dim)
        f.addRow("Feather:", self.sld_rembg_feather)
        f.addRow("Erode:", self.sld_rembg_erode)

        # Check ranges of sliders
        self._wire_percentile_guards()

        return card

    def _group_solver(self) -> QGroupBox:
        help_html = (
            "<b>General options</b><br>"
            "<u>Pins</u>: number of nails around the circle.<br>"
            "<u>Max lines</u>: number of threads to compute.<br>"
            "<u>Min distance</u>: minimum direct neighbors skipped to avoid short lines.<br>"
            "<u>Line weight</u>: how much one thread lowers residual (higher = darker, coarser).<br>"
            "<u>Cooldown last-N</u>: avoid revisiting the last N pins to reduce streaks."
        )
        card = CardGroup("General options", help_html, self)
        f = card.form

        self.sld_pins = IntSlider(12, 2048, 300)
        self.sld_steps = IntSlider(1, 20000, 4000, tick=1000)
        self.sld_min_dist = IntSlider(0, 512, 30, suffix=" pins", tick=16)
        self.sld_line_weight = FloatSlider(0.1, 16.0, 8.0, step=0.01)
        self.sld_lastn = IntSlider(0, 256, 20)

        f.addRow("Pins:", self.sld_pins)
        f.addRow("Max lines:", self.sld_steps)
        f.addRow("Min distance:", self.sld_min_dist)
        f.addRow("Line weight:", self.sld_line_weight)
        f.addRow("Cooldown last-N:", self.sld_lastn)
        return card

    def _group_preview(self) -> QGroupBox:
        help_html = (
            "<b>Rendering Preview</b><br>"
            "<u>Darken per line</u>: how much each line darkens the preview.<br>"
            "<u>Gamma</u>: display gamma.<br>"
            "<u>Line thickness</u>: 'string' width."
        )
        card = CardGroup("Preview options", help_html, self)
        f = card.form

        self.sld_alpha = FloatSlider(0.005, 0.5, 0.10, step=0.005)
        self.sld_gamma = FloatSlider(0.5, 3.0, 1.20, step=0.05)
        self.sld_thick = IntSlider(1, 5, 1)

        f.addRow("Darken per string:", self.sld_alpha)
        f.addRow("Gamma:", self.sld_gamma)
        f.addRow("Line thickness:", self.sld_thick)

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
        return card
    
    def _group_guided(self) -> QGroupBox:
        help_html = (
            "<b>Guided build</b><br>"
            "<u>Step</u>: displays what the current step is.<br>"
            "<u>Next</u>: what the next line needs to be (from → to).<br>"
            "<u>Jump to step</u>: jump to the specified step (with ENTER or 'Go').<br>"
            "<u>Prev</u>: jumps one step back.<br>"
            "<u>Next</u>: jumps one step further.<br>"
            "<u>Load Path</u>: loads the path from the specified CSV file.<br>"
            "<u>Save Session</u>: save the current guided building session, so you can continue later.<br>"
            "<u>Load Session</u>: loads a guided building session from s specified file."
        )
        card = CardGroup("Guided Build", help_html)
        f = card.form
        f.setLabelAlignment(Qt.AlignRight)

        # current step + next pin-to-pin
        self.lbl_step = QLabel("Step: - / -")
        self.lbl_next = QLabel("Next: - → -")
        f.addRow(self.lbl_step, self.lbl_next)

        # Prev/Next row
        row_nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_prev.clicked.connect(self._step_prev)
        self.btn_next.clicked.connect(self._step_next)
        row_nav.addWidget(self.btn_prev)
        row_nav.addWidget(self.btn_next)
        row_nav.addStretch(1)
        f.addRow(row_nav)

        # jump-to input
        row_step = QHBoxLayout()
        self.spin_step = QSpinBox()
        self.spin_step.setRange(0, 0)  # set real range in _setup_guide_ui()
        self.spin_step.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_step.setMinimumWidth(90)
        self.btn_step_go = QPushButton("Jump")
        self.btn_step_go.setFixedWidth(56)
        self.btn_step_go.clicked.connect(self._on_step_jump)
        self.spin_step.editingFinished.connect(self._on_step_jump)

        row_step.addWidget(QLabel("Jump to step:"))
        row_step.addWidget(self.spin_step, 0)
        row_step.addWidget(self.btn_step_go, 0)
        row_step.addStretch(1)
        f.addRow(row_step)

        # persistence row (optional; leave connected if you already have handlers)
        row_io = QHBoxLayout()
        self.btn_load_path = QPushButton("Load Path…")
        self.btn_save_session = QPushButton("Save Session…")
        self.btn_load_session = QPushButton("Load Session…")
        self.btn_load_path.clicked.connect(self._guided_load_path)
        self.btn_save_session.clicked.connect(self._guided_save_session)
        self.btn_load_session.clicked.connect(self._guided_load_session)
        row_io.addWidget(self.btn_load_path)
        row_io.addWidget(self.btn_save_session)
        row_io.addWidget(self.btn_load_session)
        f.addRow(row_io)

        return card

    # ----------- Menu -----------
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

    # ----------- Drag & Drop -----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            self.load_image(url.toLocalFile())
            break

    # ----------- Load File -----------
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

    # ----------- Run Converter -----------
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

        # guided params
        self.guided_path = path
        self.guided_pins = pins
        self.guided_work_size = self.current_work_size
        self.guided_index = -1
        self._setup_guide_ui()
        self._render_guide()

        self.btn_convert.setEnabled(True)
        self.btn_save_preview.setEnabled(bool(path))
        self.btn_export_path.setEnabled(bool(path))

        if not path:
            info(self, "Conversion", "No path produced (try other parameters).")
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
        rgb = np.dstack([preview_u8]*3)
        self.image_label.setPixmap(to_qpixmap_from_rgb(rgb, self.image_label.size()))
        info(self, "Done", f"Generated {len(path)} segments.")

    def on_errored(self, msg: str):
        self.btn_convert.setEnabled(True)
        error(self, "Error during conversion", msg)

    # ----------- Save Methods -----------
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

    # ----------- Settings Import -----------
    def _build_wmap(self):
        """
        Map parameter keys to widgets.
        """
        self.wmap = {
            # solver
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
            "pp_rembg":          self.chk_rembg,
            "pp_rembg_dim":      self.sld_rembg_dim,
            "pp_rembg_feather":  self.sld_rembg_feather,
            "pp_rembg_erode":    self.sld_rembg_erode,
        }

    def _load_presets_json(self):
        """
        Read settings.json, apply settings to widgets (ranges, default, and preset list).
        """
        # load file
        cfg_path = os.path.join(os.path.dirname(__file__), "settings.json")
        self._cfg = load_presets_json(cfg_path)

        # apply ranges to widgets
        set_widget_ranges(self._cfg.get("ranges", {}), self.wmap)

        # apply default
        default = clamp_to_ranges(self._cfg.get("default", {}), self._cfg.get("ranges", {}))
        apply_to_widgets(default, self.wmap)

        # fill preset combo
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        self.combo_preset.addItem("Default")
        for p in self._cfg.get("presets", []):
            self.combo_preset.addItem(p.get("name", "Untitled"))
        self.combo_preset.blockSignals(False)

        self.combo_preset.setCurrentIndex(0)
        self._on_preset_changed(0)

    def _on_preset_changed(self, idx: int):
        if not hasattr(self, "_cfg"):
            return
        ranges = self._cfg.get("ranges", {})
        if idx == 0:
            params = clamp_to_ranges(self._cfg.get("default", {}), ranges)
        else:
            preset = self._cfg["presets"][idx - 1]
            params = dict(self._cfg.get("default", {}))
            params.update(preset.get("params", {}))
            params = clamp_to_ranges(params, ranges)
        apply_to_widgets(params, self.wmap)

    # --------------- Batch prams get -----------------------
    def start_batch_search(self):
        if self.img_bgr is None:
            info(self, "No image", "Load an image first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder for batch results")
        if not out_dir:
            return

        base = self.gather_params()

        # This is the "grid search" for the batch search here, as you already are in the code you can use this to generate
        # multiple images/paths to find the params that fit you image the best. feel free to modify any of the ranges below.
        # It searches through all possible combinations so dont search through all of them at once :D
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

            "pp_gamma":        [0.65],
            "pp_clip_high":    [95.0],         

            "line_weight":     [8],
            "min_distance":    [base["min_distance"]],

            "pins":            [base["pins"]],
            "steps":           [base["steps"]],
            "last_n":          [base["last_n"]],
            "work_size":       [base["work_size"]],

            # preview kept constant so visual comparison is doable:
            "render_alpha":    [base["render_alpha"]],
            "render_gamma":    [base["render_gamma"]],
            "line_thickness":  [base["line_thickness"]],
        }

        def _prune_grid(base: dict, grid: dict) -> dict:
            """
            Return a copy of grid with dependent possibilities collapsed when their toggle is False.

            Example: If pp_edges includes False, collapse pp_edge_weight for the False branch.
            """
            g = {k: list(v) for k, v in grid.items()}

            if "pp_edges" in g and g["pp_edges"] == [False]:
                g["pp_edge_weight"] = [base.get("pp_edge_weight", g.get("pp_edge_weight", [0.35])[0])]

            if "pp_rembg" in g and g["pp_rembg"] == [False]:
                g["pp_rembg_dim"] = [base.get("pp_rembg_dim", g.get("pp_rembg_dim", [0.0])[0])]
                g["pp_rembg_feather"] = [base.get("pp_rembg_feather", g.get("pp_rembg_feather", [8])[0])]
                g["pp_rembg_erode"] = [base.get("pp_rembg_erode", g.get("pp_rembg_erode", [1])[0])]

            return g

        grid = _prune_grid(base, grid)

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

    # --------------- Guided methods -----------------------
    def _set_guided_enabled(self, on: bool):
        for w in (self.lbl_step, self.lbl_next,
                self.spin_step, self.btn_step_go,
                self.btn_prev, self.btn_next,
                self.btn_load_path, self.btn_save_session, self.btn_load_session):
            w.setEnabled(on)

    def _setup_guide_ui(self):
        N = len(self.guided_path)
        self.spin_step.blockSignals(True)
        self.spin_step.setRange(0, max(0, N))
        # keep current index if valid; else reset
        v = self.guided_index if 0 <= self.guided_index <= N else 0
        self.spin_step.setValue(v)
        self.spin_step.blockSignals(False)
        self._set_guided_enabled(N > 0)
        self._update_step_label()

    def _update_step_label(self):
        N = len(self.guided_path)
        i = self.guided_index
        self.lbl_step.setText(f"Step: {i + 1} / {N}")
        if 0 <= i < N:
            a, b = self.guided_path[i]
            self.lbl_next.setText(f"Next: {a} → {b}")
        else:
            self.lbl_next.setText("Next: - → -")

    def _on_step_jump(self):
        N = len(self.guided_path)
        v = int(self.spin_step.value())
        v = max(0, min(N, v))
        if v != self.guided_index:
            self.guided_index = v
            self._render_guide()
        else:
            self._update_step_label()

    def _step_prev(self):
        if self.guided_index > 0:
            self.guided_index -= 1
            self.spin_step.blockSignals(True)
            self.spin_step.setValue(self.guided_index)
            self.spin_step.blockSignals(False)
            self._render_guide()

    def _step_next(self):
        if self.guided_index < len(self.guided_path):
            self.guided_index += 1
            self.spin_step.blockSignals(True)
            self.spin_step.setValue(self.guided_index)
            self.spin_step.blockSignals(False)
            self._render_guide()


    def _guided_save_session(self):
        if not self.guided_path or self.guided_pins is None:
            info(self, "No session", "Run a conversion or load a path first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save session", "session.json", "JSON (*.json)")
        if not path:
            return
        import json
        data = {
            "work_size": self.guided_work_size,
            "index": self.guided_index,
            "pins": self.guided_pins.tolist(),
            "path": self.guided_path,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            info(self, "Saved", f"Session saved to:\n{path}")
        except Exception as e:
            error(self, "Save failed", str(e))

    def _guided_load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load session", "", "JSON (*.json)")
        if not path:
            return
        import json
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.guided_work_size = int(data["work_size"])
            self.guided_index = int(data.get("index", 0))
            self.guided_pins = np.asarray(data["pins"], dtype=np.int32)
            self.guided_path = [tuple(x) for x in data["path"]]
            self._setup_guide_ui()
            self._render_guide()
            info(self, "Loaded", f"Session loaded:\n{path}")
        except Exception as e:
            error(self, "Load failed", str(e))

    def _guided_load_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load path CSV", "", "CSV (*.csv *.txt)")
        if not path:
            return
        import csv
        try:
            segs = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                # skip header if present
                first = next(reader)
                try:
                    a, b = int(first[0]), int(first[1])
                    segs.append((a, b))
                except Exception:
                    pass  # header line; ignore
                for row in reader:
                    if len(row) < 2:
                        continue
                    segs.append((int(row[0]), int(row[1])))

            if not segs or self.current_pins is None:
                warn(self, "Load path", "No segments or no current pins available.")
                return

            self.guided_path = segs
            self.guided_pins = self.current_pins if self.current_pins is not None else self.guided_pins
            self.guided_work_size = self.current_work_size if self.current_work_size else self.guided_work_size
            self.guided_index = 0
            self._setup_guide_ui()
            self._render_guide()
            info(self, "Loaded", f"Loaded {len(segs)} segments from:\n{path}")
        except Exception as e:
            error(self, "Load failed", str(e))

    def _render_guide(self):
        if self.guided_pins is None or self.guided_work_size <= 0:
            return
        N = len(self.guided_path)
        k = max(0, min(self.guided_index, N))

        # draw segments [0..k-1]
        past_path = self.guided_path[:k]
        preview_u8 = render_path(
            work_size=self.guided_work_size,
            pins=self.guided_pins,
            path=past_path,
            alpha_per_line=self.sld_alpha.value(),
            gamma=self.sld_gamma.value(),
            thickness=self.sld_thick.value(),
        )

        # convert base grayscale to RGB
        disp = np.dstack([preview_u8] * 3)

        # circular board mask (same as in render_path)
        H = W = self.guided_work_size
        yy, xx = np.ogrid[:H, :W]
        cx, cy = W * 0.5, H * 0.5
        r = min(H, W) * 0.5 - 16
        board = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r

        # draw next segment in blue
        if k < N:
            a, b = self.guided_path[k]
            x0, y0 = int(self.guided_pins[a][0]), int(self.guided_pins[a][1])
            x1, y1 = int(self.guided_pins[b][0]), int(self.guided_pins[b][1])

            # draw directly on the RGB image
            overlay = disp.copy()
            cv2.line(
                overlay,
                (x0, y0),
                (x1, y1),
                (0, 140, 255),
                thickness=self.sld_thick.value(),
                lineType=cv2.LINE_AA,
            )

            disp[board] = overlay[board]

        self.image_label.setPixmap(to_qpixmap_from_rgb(disp, self.image_label.size()))
        self._update_step_label()

# endregion