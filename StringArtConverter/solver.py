from typing import List, Optional, Tuple
import math, numpy as np
import cv2

from StringArtConverter.utils import Segment

# -------------------- Pin + Line Precomputation --------------------

def pin_positions_circle(size: int, n_pins: int, margin: int = 1) -> np.ndarray:
    """
    Get positions of pins
    """
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

def solve_string_art(
    source_brightness_u8: np.ndarray,
    n_pins: int,
    max_lines: int,
    *,
    min_distance: int = 30,
    line_weight: float = 8.0,
    last_n: int = 20,
    work_size: int = 512,
    importance_map: Optional[np.ndarray] = None,
    progress_cb: Optional[callable] = None,
) -> Tuple[List[Segment], np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes sequence of pin-to-pin connections based on error minimization
    Parameters:
        source_brightness_u8: np.ndarray
            Target on which error is based on
        max_lines: int
            How many connections are calculated
        min_distance: int
            How close on the ring next pin can be to current one
        line_weight: float
            How much each new line contributes
        last_n: int
            Forbids the revist of a pin for a number of lines
        work_size: int
            On what resolution the error is calculated
        progress_cb: Optional[callable]
            Optional progress tracking parameter
    Returns:
        List[Segment]
            A List with all Segments of the calculated path
        np.ndarray
            Error at the end of computation
        np.ndarray
            Target used
        np.ndarray
            Pin positions used
    """

    # Ensure shape = (H,W)
    if source_brightness_u8.ndim == 1:
        gray = source_brightness_u8.reshape(work_size, work_size).astype(np.float32)
    else:
        gray = source_brightness_u8.astype(np.float32)

    H, W = gray.shape
    pins = pin_positions_circle(work_size, n_pins)
    line_cache = precalc_lines(pins, n_pins, work_size, min_distance)

    # make sure is same size
    gray = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)

    if importance_map is not None and importance_map.shape != (work_size, work_size):
        importance_map = cv2.resize(importance_map, (work_size, work_size), interpolation=cv2.INTER_NEAREST)

    # Flatten arrays for line_cache indexing
    gray = gray.astype(np.float32).ravel()
    if importance_map is None:
        importance_map = np.ones(work_size * work_size, dtype=np.float32)
    else:
        importance_map = importance_map.astype(np.float32).ravel()

    # error starts as (255 - brightness)
    #error = 255.0 - gray.ravel()
    max_fibers_per_pixel = 200.0
    target_fibers = (1.0 - gray / 255.0) * max_fibers_per_pixel

    # Importance map (default = 1 everywhere)
    if importance_map is None:
        importance_map = np.ones_like(gray, dtype=np.float32)

    # Current fiber coverage
    current_fibers = np.zeros_like(target_fibers, dtype=np.float32)

    # Quadratic error (weighted)
    error = importance_map * (target_fibers - current_fibers) ** 2

    path: List[Segment] = []
    current_pin = 0
    last_pins = [-1] * last_n

    for step in range(max_lines):
        if progress_cb:
            progress_cb(int(100 * step / max(1, max_lines - 1)))

        best_pin = None
        best_gain = -1e9
        best_idx = None

        for offset in range(min_distance, n_pins - min_distance):
            test_pin = (current_pin + offset) % n_pins
            if test_pin in last_pins:
                continue
            idx = line_cache.get((current_pin, test_pin))
            if idx is None:
                continue

            old_err = error[idx].sum()

            new_fibers = current_fibers[idx] + line_weight
            new_err = (importance_map.ravel()[idx] *
                       (target_fibers.ravel()[idx] - new_fibers) ** 2).sum()

            gain = old_err - new_err
            if gain > best_gain:
                best_gain = gain
                best_pin = test_pin
                best_idx = idx

        if best_pin is None:
            break   # nothing improves anymore

        # Apply line: subtract constant LINE_WEIGHT along the pixels
        current_fibers[best_idx] += line_weight
        error[best_idx] = (importance_map.ravel()[best_idx] *
                           (target_fibers.ravel()[best_idx] - current_fibers[best_idx]) ** 2)

        path.append((current_pin, best_pin))
        last_pins.append(best_pin)
        last_pins = last_pins[-last_n:]
        current_pin = best_pin

    if progress_cb:
        progress_cb(100)

    return path, error.reshape(H, W), gray, pins