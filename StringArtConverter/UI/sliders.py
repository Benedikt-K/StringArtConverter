from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider

class NoWheelSlider(QSlider):
    def wheelEvent(self, e):
        e.ignore()

class IntSlider(QWidget):
    """
    Int slider that has a step size and min/max
    """
    valueChanged = Signal(int)

    def __init__(self, minimum: int, maximum: int, value: int, *, suffix: str = "", tick: int | None = None, parent=None):
        super().__init__(parent)
        self._suffix = suffix
        self._slider = NoWheelSlider(Qt.Horizontal)
        self._label = QLabel(f"{value}{suffix}")
        self._label.setMinimumWidth(72)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._label, 0)

        self._slider.setRange(minimum, maximum)
        self._slider.setValue(value)
        if tick:
            self._slider.setTickInterval(int(tick))
            self._slider.setTickPosition(QSlider.TicksBelow)
        
        self._slider.valueChanged.connect(self._on_change)

    def setRange(self, minimum: int, maximum: int):
        self._slider.setRange(int(minimum), int(maximum))
        self.setValue(self.value())

    def set_step(self, step: int):
        self._slider.setPageStep(max(1, int(step)))

    def _on_change(self, v: int):
        self._label.setText(f"{v}{self._suffix}")
        self.valueChanged.emit(v)

    def value(self) -> int:
        return int(self._slider.value())

    def setValue(self, v: int):
        self._slider.setValue(int(v))

    def setEnabled(self, e: bool):
        self._slider.setEnabled(e)
        self._label.setEnabled(e)


class FloatSlider(QWidget):
    """
    Float slider with fixed precision using an internal integer scale.
    Example: FloatSlider(0.0, 1.0, 0.35, step=0.01) -> shows 2 decimals
    """
    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, value: float, *, step: float = 0.01, decimals: int | None = None,
                 suffix: str = "", tick: float | None = None, parent=None):
        super().__init__(parent)
        self._decimals = decimals if decimals is not None else max(0, len(str(step).split(".")[-1]))
        self._suffix = suffix
        self._slider = NoWheelSlider(Qt.Horizontal)
        self._label = QLabel(self._fmt(value))
        self._label.setMinimumWidth(72)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._label, 0)

        self.set_step(step)
        self._slider.setRange(minimum, maximum)
        self._slider.setValue(value)
        if tick:
            self._slider.setTickInterval(int(round(tick * self._scale)))
            self._slider.setTickPosition(QSlider.TicksBelow)

        self._slider.valueChanged.connect(self._on_change)

    def set_step(self, step: float):
        self._scale = int(round(1.0 / max(1e-9, float(step))))
        if hasattr(self, "_min") and hasattr(self, "_max"):
            self._slider.setRange(0, int(round((self._max - self._min) * self._scale)))
            self.setValue(self.value())

    def set_range(self, minimum: float, maximum: float):
        self._min = float(minimum)
        self._max = float(maximum)
        self._slider.setRange(0, int(round((self._max - self._min) * self._scale)))
        self.setValue(self.value())

    def setRange(self, minimum: float, maximum: float):
        self.set_range(minimum, maximum)

    def _fmt(self, v: float) -> str:
        return f"{v:.{self._decimals}f}{self._suffix}"

    def _on_change(self, raw: int):
        v = self._min + raw / self._scale
        v = max(self._min, min(self._max, v))
        self._label.setText(self._fmt(v))
        self.valueChanged.emit(v)

    def value(self) -> float:
        raw = self._slider.value()
        return self._min + raw / self._scale

    def setValue(self, v: float):
        v = max(self._min, min(self._max, v))
        self._slider.setValue(int(round((v - self._min) * self._scale)))

    def setEnabled(self, e: bool):
        self._slider.setEnabled(e)
        self._label.setEnabled(e)