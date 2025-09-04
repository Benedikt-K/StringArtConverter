from typing import Tuple, List
import json

Segment = Tuple[int, int]

def save_path_txt(path: List[Segment], out_txt: str) -> None:
    """
    Save path to file
    """
    with open(out_txt, "w", encoding="utf-8") as f:
        for a, b in path:
            f.write(f"{a} {b}\n")

def load_presets_json(path: str) -> dict:
    """
    Load preset values, ranges, presets from json file
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("ranges", {})
    data.setdefault("defaults", {})
    data.setdefault("presets", [])
    return data

def clamp_to_ranges(params: dict, ranges: dict) -> dict:
    """
    Ensure values are in range
    """
    out = dict(params)
    for k, r in ranges.items():
        if k in out and isinstance(out[k], (int, float)):
            lo = r.get("min", out[k]); hi = r.get("max", out[k])
            out[k] = max(lo, min(hi, out[k]))
    return out
