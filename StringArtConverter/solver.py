from typing import List, Optional, Tuple
import math, numpy as np


Segment = Tuple[int, int]

# -------------------- Pin + Line Precomputation --------------------

def pin_positions_circle(size: int, n_pins: int, margin: int = 1) -> np.ndarray:
    cx, cy = size / 2, size / 2
    r = size / 2 - margin
    ang = np.linspace(0, 2 * math.pi, n_pins, endpoint=False)
    xs = (cx + r * np.cos(ang)).astype(np.int32)
    ys = (cy + r * np.sin(ang)).astype(np.int32)
    return np.stack([xs, ys], axis=1)

def precalc_lines(pins: np.ndarray, n_pins: int, size: int, min_distance: int):
    """
    Precompute pixel indices for each line (both directions).
    """
    H = W = size
    line_cache = {}
    for i in range(n_pins):
        for j in range(i + min_distance, n_pins):
            x0, y0 = pins[i]
            x1, y1 = pins[j]
            d = int(math.hypot(x1 - x0, y1 - y0))
            if d <= 1: 
                continue
            xs = np.linspace(x0, x1, d).astype(np.int32)
            ys = np.linspace(y0, y1, d).astype(np.int32)
            idx = ys * W + xs
            line_cache[(i, j)] = idx
            line_cache[(j, i)] = idx
    return line_cache

# -------------------- Core Solver --------------------

def solve_string_art_go(
    source_brightness_u8: np.ndarray,   # <-- preprocessed brightness (0..255), 2D or flat
    n_pins: int,
    max_lines: int,
    *,
    min_distance: int = 30,
    line_weight: float = 8.0,
    last_n: int = 20,
    work_size: int = 500,
    progress_cb: Optional[callable] = None,
) -> Tuple[List[Segment], np.ndarray, np.ndarray, np.ndarray]:

    # Ensure shape = (H,W)
    if source_brightness_u8.ndim == 1:
        gray = source_brightness_u8.reshape(work_size, work_size).astype(np.float32)
    else:
        gray = source_brightness_u8.astype(np.float32)

    H, W = gray.shape
    pins = pin_positions_circle(work_size, n_pins)
    line_cache = precalc_lines(pins, n_pins, work_size, min_distance)

    # Go logic: error starts as (255 - brightness)
    error = 255.0 - gray.ravel()

    path: List[Segment] = []
    current_pin = 0
    last_pins = [-1] * last_n

    for step in range(max_lines):
        if progress_cb:
            progress_cb(int(100 * step / max(1, max_lines - 1)))

        best_pin = None
        best_err = -1.0
        best_idx = None

        for offset in range(min_distance, n_pins - min_distance):
            test_pin = (current_pin + offset) % n_pins
            if test_pin in last_pins:
                continue
            idx = line_cache.get((current_pin, test_pin))
            if idx is None:
                continue

            line_err = error[idx].sum()
            if line_err > best_err:
                best_err = line_err
                best_pin = test_pin
                best_idx = idx

        if best_pin is None:
            break

        # Apply line: subtract constant LINE_WEIGHT along the pixels
        error[best_idx] -= float(line_weight)
        np.maximum(error, 0.0, out=error)

        path.append((current_pin, best_pin))
        last_pins.append(best_pin)
        last_pins = last_pins[-last_n:]
        current_pin = best_pin

    if progress_cb:
        progress_cb(100)

    # Return: path, final error map, the brightness target we used, and pins
    return path, error.reshape(H, W), gray, pins