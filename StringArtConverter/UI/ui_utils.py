import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QColor
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QComboBox,
    QDoubleSpinBox, QHBoxLayout, QToolButton, QToolTip, QGraphicsDropShadowEffect
)
from StringArtConverter.UI.sliders import IntSlider, FloatSlider

class ClickableLabel(QLabel):
    """
    QLabel subclass that emits a signal when clicked.
    """
    clicked = Signal()

    def mouseReleaseEvent(self, e):
        """
        Emit 'clicked' signal when the label is left-clicked.
        """
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)

class NonScrollComboBox(QComboBox):
    """
    QComboBox that ignores scroll-wheel events to prevent accidental changes.
    """
    def wheelEvent(self, event):
        event.ignore()

class HelpBadge(QToolButton):
    """
    Small circular help button that shows a tooltip on click or hover.
    """
    def __init__(self, tooltip_html: str, parent=None):
        """
        Args:
            tooltip_html (str): HTML content displayed as tooltip.
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)
        self.setText("?")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip_html)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setFixedSize(22, 22)
        self.setObjectName("HelpBadge")
        self.clicked.connect(self._show_tooltip_now)

    def _show_tooltip_now(self):
        """
        Show the tooltip at the current cursor position.
        """
        QToolTip.showText(QCursor.pos(), self.toolTip(), self)

class CardGroup(QWidget):
    """
    Card-style widget containing a titled header and form area.
    """
    def __init__(self, title: str, help_html: str, parent=None):
        """
        Args:
            title (str): Title text displayed in the card header.
            help_html (str): HTML help text displayed in the tooltip badge.
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)
        self.setObjectName("CardGroup")
        self.setAttribute(Qt.WA_StyledBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Header row
        hdr = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setObjectName("CardTitle")
        lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        hdr.addWidget(lbl, 1)

        badge = HelpBadge(help_html)
        hdr.addWidget(badge, 0, Qt.AlignRight | Qt.AlignVCenter)
        root.addLayout(hdr)

        # Body
        self.form = QFormLayout()
        self.form.setLabelAlignment(Qt.AlignRight)
        root.addLayout(self.form)

def add_card_shadow(w):
    """
    Apply a soft shadow effect to a widget.

    Args:
        w (QWidget): Target widget to which the shadow effect is applied.
    """
    fx = QGraphicsDropShadowEffect(w)
    fx.setBlurRadius(18)
    fx.setOffset(0, 6)
    fx.setColor(QColor(0, 0, 0, 110))
    w.setGraphicsEffect(fx)

def apply_to_widgets(params: dict, wmap: dict):
    """
    Set widget values based on a parameter dictionary.

    Args:
        params (dict): Mapping of parameter names to values.
        wmap (dict): Mapping of widget names to their corresponding instances.
    """
    for key, val in params.items():
        widget = wmap.get(key)
        if widget is None:
            continue
        if hasattr(widget, "setChecked") and isinstance(val, (bool, np.bool_)):
            widget.setChecked(bool(val))
        elif hasattr(widget, "setValue"):
            widget.setValue(val)

def set_widget_ranges(ranges: dict, wmap: dict):
    """
    Configure min/max/step ranges input widgets.

    Supports QSpinBox, QDoubleSpinBox, IntSlider, and FloatSlider.

    Args:
        ranges (dict): Mapping of widget names to range specifications. Each value 
            should be a dict with optional keys: 'min', 'max', and 'step'.
        wmap (dict): Mapping of widget names to widget instances.
    """
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

        if isinstance(w, IntSlider):
            if lo is not None and hi is not None:
                w.setRange(int(lo), int(hi))
            if step is not None:
                w.setStep(int(step))
            continue

        if isinstance(w, FloatSlider):
            if lo is not None and hi is not None:
                if hasattr(w, "set_range"):
                    w.set_range(float(lo), float(hi))
                else:
                    w.setRange(float(lo), float(hi))
            if step is not None and hasattr(w, "set_step"):
                w.set_step(float(step))
            continue

def collect_from_widgets(wmap: dict) -> dict:
    """
    Read and collect values from a group of widgets into a dictionary.

    Args:
        wmap (dict): Mapping of parameter names to widget instances.

    Returns:
        dict: Dictionary mapping widget names to their current values.
    """
    out = {}
    for key, widget in wmap.items():
        if hasattr(widget, "isChecked"):
            out[key] = bool(widget.isChecked())
        elif hasattr(widget, "value"):
            out[key] = widget.value()
    return out