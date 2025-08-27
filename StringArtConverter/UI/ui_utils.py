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

def apply_to_widgets(params: dict, wmap: dict):
    """wmap: {param_name: widget}"""
    for name, w in wmap.items():
        if name not in params:
            continue
        val = params[name]
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.setValue(val)
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(val))
        elif isinstance(w, QComboBox):
            # if map to discrete strings, handle here
            idx = w.findText(str(val))
            if idx >= 0: w.setCurrentIndex(idx)

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
    for name, w in wmap.items():
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            out[name] = w.value()
        elif isinstance(w, QCheckBox):
            out[name] = w.isChecked()
        elif isinstance(w, QComboBox):
            out[name] = w.currentText()
    return out
