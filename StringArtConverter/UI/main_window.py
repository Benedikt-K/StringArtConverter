# region ----------- imports -----------
from __future__ import annotations
from typing import List, Optional
import math
import cv2
import numpy as np
import os
import csv

# -------- UI ----------
from PySide6.QtCore import Qt, QThread, QSize
from PySide6.QtGui import QAction, QPixmap, QImage
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QCheckBox,QGroupBox, QProgressBar, 
    QMessageBox, QScrollArea, QFrame, QComboBox, QHBoxLayout, QSpinBox
)
from StringArtConverter.UI.sliders import IntSlider, FloatSlider
from StringArtConverter.UI.ui_utils import (
    ClickableLabel, CardGroup, NonScrollComboBox,
    apply_to_widgets, set_widget_ranges, add_card_shadow
)
# -------- worker --------
from StringArtConverter.UI.workers import ConvertWorker, BatchSearchWorker

# -------- solver  --------
from StringArtConverter.utils import load_presets_json, clamp_to_ranges, Segment
from StringArtConverter.previewer import render_path
from StringArtConverter.solver import pin_positions_circle

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

        self.is_startup: bool = True

        self._thread: Optional[QThread] = None
        self._worker: Optional[ConvertWorker] = None

        self._build_ui()
        self._build_menu()

        self.guided_path: list[tuple[int, int]] = []
        self.guided_pins: Optional[np.ndarray] = None
        self.guided_work_size: int = 0
        self.guided_index: int = -1
        self.is_render_guided: bool = False

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

        # params search button - currently disabled
        self.btn_batch = QPushButton("Batch Preset Search…")
        self.btn_batch.clicked.connect(self.start_batch_search)
        self.btn_batch.setEnabled(False)

        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        run_row.addWidget(self.btn_convert, 0)
        #run_row.addWidget(self.btn_batch, 0)
        run_row.addWidget(self.progress, 1)
        right_layout.addLayout(run_row)
        right_layout.addStretch(1)
        right_layout.setAlignment(Qt.AlignTop)

        # preview
        self.group_preview = self._group_preview()
        right_layout.addWidget(self.group_preview)

        # presets
        self.combo_preset = NonScrollComboBox()
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
            "<u>Work size</u>: 'Canvas Size' (higher = enables more lines, but gets slower).<br>"
            "<u>CLAHE</u>: local contrast equalization (can add noise).<br>"
            "<u>Contrast stretch</u>: remap dark/bright percentiles ('compress' grayscale values).<br>"
            "<u>Blend edges</u>: mix edges into the target (higher = more focused on contours).<br>"
            "<u>Face detection</u>: give more weight to faces.<br>"
            "<u>Darken background (rembg)</u>: darken background by the given dim factor;"
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

        # Masking options
        self.chk_rembg = QCheckBox("Darken background")
        self.sld_rembg_dim = FloatSlider(0.0, 1.0, 0.6, step=0.05)
        self.sld_rembg_feather = IntSlider(0, 64, 8, suffix=" px", tick=4)
        self.sld_rembg_erode = IntSlider(0, 8, 1, suffix=" px")
        self.chk_face_fg_detection = QCheckBox("Face detection")
        f.addRow(self.chk_face_fg_detection)
        f.addRow(self.chk_rembg)
        f.addRow("Dim factor:", self.sld_rembg_dim)

        # Check ranges of sliders
        self._wire_percentile_guards()

        return card

    def _group_solver(self) -> QGroupBox:
        help_html = (
            "<b>General options</b><br>"
            "<u>Pins</u>: number of nails around the circle.<br>"
            "<u>Number of Lines</u>: number of threads to compute.<br>"
            "<u>Min distance</u>: minimum direct neighbors skipped to avoid short lines.<br>"
            "<u>Line weight</u>: how much one thread influences the residual "
            "(higher = faster convergence, but less detail later on).<br>"
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
        f.addRow("Number of Lines:", self.sld_steps)
        f.addRow("Min distance:", self.sld_min_dist)
        f.addRow("Line weight:", self.sld_line_weight)
        f.addRow("Cooldown last-N:", self.sld_lastn)
        return card

    def _group_preview(self) -> QGroupBox:
        help_html = (
            "<b>Rendering Preview</b><br>"
            "<u>Darken per string</u>: how much each line darkens the preview - relates to opacity/color of the chosen string (gray, black...).<br>"
            "<u>Gamma</u>: gamma of the displayed render.<br>"
            "<u>Line thickness</u>: 'string' width.<br>"
            "<u>Save Preview</u>: saves the generated preview image<br>"
            "<u>Save Path</u>: saves the generated sequence of pins as a csv file."
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
        self.btn_export_path = QPushButton("Save Path…")
        self.btn_export_path.clicked.connect(self.export_path)
        self.btn_export_path.setEnabled(False)
        row.addWidget(self.btn_save_preview)
        row.addWidget(self.btn_export_path)
        f.addRow(row)
        return card
    
    def _group_guided(self) -> QGroupBox:
        help_html = (
            "<b>Pin-by-pin build</b><br>"
            "<u>Step</u>: displays what the current step is.<br>"
            "<u>Next</u>: what the next line needs to be (from → to).<br>"
            "<u>Prev</u>: jumps one step back.<br>"
            "<u>Next</u>: jumps one step further.<br>"
            "<u>Switch Preview</u>: Switches the preview between the completed image and the current pin-by-pin step.<br>"
            "<u>Jump to step</u>: jump to the specified step (with ENTER or 'Jump').<br>"
            "<u>Load Path</u>: loads the path from the specified CSV file."
        )
        card = CardGroup("Guided Build", help_html)
        f = card.form
        f.setLabelAlignment(Qt.AlignRight)

        # current step + next pin-to-pin + switch preview
        row_upper = QHBoxLayout()
        self.lbl_step = QLabel("Step: - / -")
        self.lbl_next = QLabel("Next: - → -")
        self.lbl_switch = QLabel("Finished | Pin-by-Pin")
        row_upper.addWidget(self.lbl_step)
        row_upper.addWidget(self.lbl_next)
        row_upper.addStretch()
        row_upper.addWidget(self.lbl_switch)
        f.addRow(row_upper)

        # Prev/Next/switch row buttons
        row_nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀  Prev")
        self.btn_next = QPushButton("Next  ▶")
        self.btn_switch = QPushButton("  Switch Preview  ")
        self.btn_prev.clicked.connect(self._step_prev)
        self.btn_next.clicked.connect(self._step_next)
        self.btn_switch.clicked.connect(self._switch_preview)
        row_nav.addWidget(self.btn_prev)
        row_nav.addWidget(self.btn_next)
        row_nav.addStretch()
        row_nav.addWidget(self.btn_switch)
        f.addRow(row_nav)

        # jump-to input
        row_step = QHBoxLayout()
        self.spin_step = QSpinBox()
        self.spin_step.setRange(0, 0)  # set real range in _setup_guide_ui()
        self.spin_step.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_step.setMinimumWidth(90)
        self.btn_jump_to = QPushButton("Jump")
        self.btn_jump_to.setFixedWidth(56)
        self.btn_jump_to.clicked.connect(self._on_step_jump)
        self.spin_step.editingFinished.connect(self._on_step_jump)

        row_step.addWidget(QLabel("Jump to step:"))
        row_step.addWidget(self.spin_step, 0)
        row_step.addWidget(self.btn_jump_to, 0)
        row_step.addStretch(1)
        f.addRow(row_step)

        # save/load guided session
        row_io = QHBoxLayout()
        self.btn_load_path = QPushButton("Load Path…")
        self.btn_save_session = QPushButton("Save Session…")
        self.btn_load_session = QPushButton("Load Last Session…")
        self.btn_load_path.clicked.connect(self._guided_load_path)
        self.btn_save_session.clicked.connect(self._guided_save_session)
        self.btn_load_session.clicked.connect(self._guided_load_session)
        row_io.addWidget(self.btn_load_path)
        #row_io.addWidget(self.btn_save_session)
        #row_io.addWidget(self.btn_load_session)

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
            pp_face_mask=self.chk_face_fg_detection.isChecked(),
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
        self._render_guided()

        self.btn_convert.setEnabled(True)
        self.btn_save_preview.setEnabled(bool(path))
        self.btn_export_path.setEnabled(bool(path))
        self.lbl_switch.setText('<span style="font-weight:bold;">Finished</span> | Pin-by-Pin')

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
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                
                # metadata header
                num_pins = len(self.current_pins) if self.current_pins is not None else 0
                work_size = self.current_work_size if self.current_work_size is not None else 0
                w.writerow([f"# pins={num_pins}", f"work size={work_size}"])

                # data header
                w.writerow(["from_pin", "to_pin"])
                
                # data (path)
                w.writerows(self.current_path)
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
            "pp_face_mask":      self.chk_face_fg_detection,
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

        # apply default (balanced portrait)
        balanced_portrait = clamp_to_ranges(self._cfg.get("default", {}), self._cfg.get("ranges", {}))
        apply_to_widgets(balanced_portrait, self.wmap)

        # fill preset combo
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        self.combo_preset.addItem("Portrait: Balanced")
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
        for w in (self.lbl_step, self.lbl_next, self.lbl_switch,
                self.spin_step, self.btn_jump_to,
                self.btn_prev, self.btn_next, self.btn_switch,
                self.btn_save_session, self.btn_load_session):
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
            self._render_guided()
        else:
            self._update_step_label()

    def _step_prev(self):
        if self.guided_index > 0:
            self.guided_index -= 1
            self.spin_step.blockSignals(True)
            self.spin_step.setValue(self.guided_index)
            self.spin_step.blockSignals(False)
            self._render_guided()

    def _step_next(self):
        if self.guided_index < len(self.guided_path):
            self.guided_index += 1
            self.spin_step.blockSignals(True)
            self.spin_step.setValue(self.guided_index)
            self.spin_step.blockSignals(False)
            self._render_guided()

    def _switch_preview(self):
        """
        Switch between rendering Full preview image and guided building image
        """
        if self.is_render_guided:
            self.is_render_guided = False
            display = render_path(
                work_size=self.current_work_size,
                pins=self.current_pins,
                path=self.current_path,
                alpha_per_line=self.sld_alpha.value(),
                gamma=self.sld_gamma.value(),
                thickness=self.sld_thick.value(),
            )
            self.image_label.setPixmap(to_qpixmap_from_rgb(display, self.image_label.size()))
            self.lbl_switch.setText('<span style="font-weight:bold;">Finished</span> | Pin-by-Pin')
        else:
            self.is_render_guided = True
            self._render_guided()
            self.lbl_switch.setText('Finished | <span style="font-weight:bold;">Pin-by-Pin</span>')

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
            self._render_guided()
            info(self, "Loaded", f"Session loaded:\n{path}")
        except Exception as e:
            error(self, "Load failed", str(e))

    def _guided_load_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load path CSV", "", "CSV (*.csv *.txt)")
        if not path:
            return
        try:
            segs = []
            num_pins = None
            work_size = None
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)

                first = next(reader)
                if first:
                    if first[0].startswith("# pins="):
                        try:
                            num_pins = int(first[0].split("=")[1])
                        except ValueError:
                            pass
                    if first[1].startswith('work size='):
                        try:
                            work_size = int(first[1].split("=")[1])
                        except ValueError:
                            pass
                    next(reader)
                else:
                    try:
                        a, b = int(first[0]), int(first[1])
                        segs.append((a, b))
                    except Exception:
                        pass  # not valid row
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        segs.append((int(row[0]), int(row[1])))
                    except ValueError:
                        continue

            if not segs:
                warn(self, "Load path", "No segments available.")
                return

            pins = pin_positions_circle(work_size, num_pins)

            self.guided_path = segs
            self.guided_pins = pins
            self.guided_work_size = work_size
            self.guided_index = -1
            self.is_render_guided = False

            # setup rest, if done directly after startup
            if (self.is_startup):
                self.current_path = segs
                self.current_pins = pins
                self.current_work_size = work_size

                params = self.gather_params()
                preview_u8 = render_path(
                    work_size=self.current_work_size,
                    pins=self.current_pins,
                    path=self.current_path,
                    alpha_per_line=self.sld_alpha.value(),
                    gamma=self.sld_gamma.value(),
                    thickness=self.sld_thick.value(),
                )
                rgb = np.dstack([preview_u8] * 3)
                self.image_label.setPixmap(to_qpixmap_from_rgb(rgb, self.image_label.size()))
                self._update_step_label()

            self._setup_guide_ui()

            # render guided and change switch button
            self.is_render_guided = True
            self._render_guided()
            self.lbl_switch.setText('Finished | <span style="font-weight:bold;">Pin-by-Pin</span>')

            info(self, "Loaded", f"Loaded {len(segs)} segments from:\n{path}")
        except Exception as e:
            error(self, "Load failed", str(e))

    def _render_guided(self):
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