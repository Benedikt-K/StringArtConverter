from typing import Tuple, List
import json

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
