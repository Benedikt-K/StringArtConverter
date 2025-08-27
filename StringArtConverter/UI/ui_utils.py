import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize
from PySide6.QtGui import QAction, QPixmap, QImage, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QProgressBar, QMessageBox, QScrollArea, QFrame, QComboBox, QHBoxLayout
)

class ClickableLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)

def apply_to_widgets(wmap: dict, params: dict):
    for key, widget in wmap.items():
        if key not in params:
            continue
        val = params[key]
        # checkboxes
        if hasattr(widget, "setChecked") and isinstance(val, (bool, np.bool_)):
            widget.setChecked(bool(val))
        # generic numeric widgets (spinboxes, sliders, custom ones)
        elif hasattr(widget, "setValue"):
            widget.setValue(val)

def set_widget_ranges(ranges: dict, wmap: dict):
    for name, spec in ranges.items():
        w = wmap.get(name)
        if w is None:
            continue
        if isinstance(w, QSpinBox):
            w.setRange(int(spec.get("min", w.minimum() or 0)),
                       int(spec.get("max", w.maximum() or 999999)))
            step = int(spec.get("step", 1))
            w.setSingleStep(step)
        elif isinstance(w, QDoubleSpinBox):
            w.setRange(float(spec.get("min", w.minimum())),
                       float(spec.get("max", w.maximum())))
            step = float(spec.get("step", w.singleStep()))
            w.setSingleStep(step)

def collect_from_widgets(wmap: dict) -> dict:
    out = {}
    for key, widget in wmap.items():
        if hasattr(widget, "isChecked"):
            out[key] = bool(widget.isChecked())
        elif hasattr(widget, "value"):
            out[key] = widget.value()
    return out