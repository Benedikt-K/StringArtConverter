from typing import Tuple, List
import json

Segment = Tuple[int, int]

def save_path_txt(path: List[Segment], out_txt: str) -> None:
    """
    Save a string art path to a text file.

    Each line of the output file contains a pair of integers representing
    a segment between two pins.

    Args:
        path (List[Tuple[int, int]]): List of (start_pin, end_pin) pairs.
        out_txt (str): Output file path for saving the path data.
    """
    with open(out_txt, "w", encoding="utf-8") as f:
        for a, b in path:
            f.write(f"{a} {b}\n")

def load_presets_json(path: str) -> dict:
    """
    Load the preset configuration values from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: Dictionary with keys "ranges", "defaults", and "presets".

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
        FileNotFoundError: If the file does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("ranges", {})
    data.setdefault("defaults", {})
    data.setdefault("presets", [])
    return data

def clamp_to_ranges(params: dict, ranges: dict) -> dict:
    """
    Clamp numeric parameters to defined minimum/maximum ranges.

    Args:
        params (dict): Dictionary of parameter names and their current values.
        ranges (dict): Dictionary of parameter names and "min" and "max".
            If a limit is missing, the value is not clamped in that direction.

    Returns:
        dict: A new dictionary with clamped parameter values.
    """
    out = dict(params)
    for k, r in ranges.items():
        if k in out and isinstance(out[k], (int, float)):
            lo = r.get("min", out[k]); hi = r.get("max", out[k])
            out[k] = max(lo, min(hi, out[k]))
    return out
