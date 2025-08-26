from typing import Tuple, List
import json, os
from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox

# def Segment
Segment = Tuple[int, int]

# ----------- save path -------------
def save_path_txt(path: List[Segment], out_txt: str) -> None:
    """
    Save path to file
    """
    with open(out_txt, "w", encoding="utf-8") as f:
        for a, b in path:
            f.write(f"{a} {b}\n")

# ----------- json settings -------------
def load_presets_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("ranges", {})
    data.setdefault("defaults", {})
    data.setdefault("presets", [])
    return data

def clamp_to_ranges(params: dict, ranges: dict) -> dict:
    out = dict(params)
    for k, r in ranges.items():
        if k in out and isinstance(out[k], (int, float)):
            lo = r.get("min", out[k]); hi = r.get("max", out[k])
            out[k] = max(lo, min(hi, out[k]))
    return out

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
            # if your combo maps to discrete strings, handle here
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