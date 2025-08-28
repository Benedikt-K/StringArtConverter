import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize
from PySide6.QtGui import QAction, QPixmap, QImage, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QProgressBar, QMessageBox, QScrollArea, QFrame, QComboBox, QHBoxLayout
)
from StringArtConverter.UI.sliders import IntSlider, FloatSlider

class ClickableLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)

def apply_to_widgets(params: dict, wmap: dict):
    for key, val in params.items():
        widget = wmap.get(key)
        if widget is None:
            continue
        if hasattr(widget, "setChecked") and isinstance(val, (bool, np.bool_)):
            widget.setChecked(bool(val))
        elif hasattr(widget, "setValue"):
            widget.setValue(val)

def set_widget_ranges(ranges: dict, wmap: dict):
    for name, spec in ranges.items():
        w = wmap.get(name)
        if w is None:
            continue

        lo = spec.get("min", None)
        hi = spec.get("max", None)
        step = spec.get("step", None)

        if isinstance(w, QSpinBox):
            if lo is not None: w.setMinimum(int(lo))
            if hi is not None: w.setMaximum(int(hi))
            if step is not None: w.setSingleStep(int(step))
            continue

        if isinstance(w, QDoubleSpinBox):
            if lo is not None: w.setMinimum(float(lo))
            if hi is not None: w.setMaximum(float(hi))
            if step is not None: w.setSingleStep(float(step))
            continue

        # Custom sliders
        if isinstance(w, IntSlider):
            if lo is not None and hi is not None:
                w.setRange(int(lo), int(hi))
            if step is not None and hasattr(w, "set_step"):
                w.set_step(int(step))
            continue

        if isinstance(w, FloatSlider):
            if lo is not None and hi is not None:
                # supports either set_range or setRange
                if hasattr(w, "set_range"):
                    w.set_range(float(lo), float(hi))
                else:
                    w.setRange(float(lo), float(hi))
            if step is not None and hasattr(w, "set_step"):
                w.set_step(float(step))
            continue

def collect_from_widgets(wmap: dict) -> dict:
    out = {}
    for key, widget in wmap.items():
        if hasattr(widget, "isChecked"):
            out[key] = bool(widget.isChecked())
        elif hasattr(widget, "value"):
            out[key] = widget.value()
    return out