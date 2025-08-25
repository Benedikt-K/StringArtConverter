from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional
import math
import cv2
import numpy as np

Segment = Tuple[int, int]

# -------------------- Simple preprocessing --------------------

def _resize_square(img_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

def _to_grayscale(img_bgr: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray

def _build_target(gray_u8: np.ndarray, edge_weight: float = 0.5) -> np.ndarray:
    """Combine darkness + edges into a residual target in [0,1]."""
    gray = gray_u8.astype(np.float32) / 255.0
    darkness = 1.0 - gray
    med = float(np.median(gray_u8))
    t1 = int(max(0, 0.66 * med))
    t2 = int(min(255, 1.33 * med) + 1)
    edges = cv2.Canny(gray_u8, t1, t2).astype(np.float32) / 255.0
    target = (1.0 - edge_weight) * darkness + edge_weight * edges
    return np.clip(target, 0.0, 1.0)

def _circular_mask(h: int, w: int, margin: int = 16) -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    cx, cy = w * 0.5, h * 0.5
    r = min(h, w) * 0.5 - margin
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (r * r)
    return mask.astype(np.float32)

# -------------------- Pins & line sampling --------------------

def _pin_positions_circle(h: int, w: int, n_pins: int, margin: int = 16) -> np.ndarray:
    """Nx2 int array of (x,y) pin coords around a circle."""
    cx, cy = w / 2.0, h / 2.0
    r = min(h, w) / 2.0 - margin
    ang = np.linspace(0.0, 2.0 * math.pi, n_pins, endpoint=False)
    xs = (cx + r * np.cos(ang)).round().astype(np.int32)
    ys = (cy + r * np.sin(ang)).round().astype(np.int32)
    return np.stack([xs, ys], axis=1)

def _bresenham_indices(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Return flat indices (int32) of a 1-px Bresenham line from (x0,y0) to (x1,y1)."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    xs, ys = [], []
    while True:
        if 0 <= x < w and 0 <= y < h:
            xs.append(x); ys.append(y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    if not xs:
        return np.empty(0, dtype=np.int32)
    return (np.asarray(ys, dtype=np.int32) * w + np.asarray(xs, dtype=np.int32))

# -------------------- Public API used by your UI --------------------

def solve_string_art(
    img_bgr: np.ndarray,
    n_pins: int,
    steps: int,
    min_hop: int = 6,
    *,
    work_size: int = 512,
    edge_weight: float = 0.5,
    draw_strength: float = 0.10,
    target_threshold: float = 0.06,
    min_steps_to_take: int = 700,
    candidate_budget: int = 64,
    blur_sigma: float = 0.8,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Tuple[List[Segment], np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy string-art solver (fast & simple).
    Returns: (path, residual, target, pins)
    - progress_cb: optional function receiving 0..100 ints (safe for your QThread signal)
    """
    if img_bgr is None or img_bgr.size == 0:
        return [], np.zeros((work_size, work_size), np.float32), np.zeros((work_size, work_size), np.float32), np.zeros((0,2), np.int32)

    # ---- Preprocess
    img_bgr = _resize_square(img_bgr, work_size)
    gray = _to_grayscale(img_bgr, use_clahe=True)
    target = _build_target(gray, edge_weight=edge_weight)

    mask = _circular_mask(*target.shape, margin=16)
    target *= mask

    residual = target.astype(np.float32).copy()
    h, w = residual.shape
    rflat = residual.ravel()

    pins = _pin_positions_circle(h, w, n_pins, margin=16).astype(np.int32)

    # ---- Candidate ring (~candidate_budget per step)
    stride = max(1, n_pins // max(1, candidate_budget))
    ring = list(range(0, n_pins, stride)) or list(range(n_pins))
    ring_len = len(ring)

    path: List[Segment] = []
    current = 0
    error = float(residual.mean())
    draw_strength = float(np.clip(draw_strength, 0.01, 1.0))

    for step in range(steps):
        if progress_cb:
            # progress by error reduction (feels better than linear time)
            progress_cb(int(100 * (1.0 - min(1.0, error))))

        if (error < target_threshold) and (step >= min_steps_to_take):
            break

        # Multi-scale once per step
        low = cv2.GaussianBlur(residual, (0, 0), blur_sigma)
        hi = residual - low
        score_field = 0.8 * low + 0.2 * hi
        sflat = score_field.ravel()

        best_j, best_score = None, -1e9

        # rotate candidate ring around the current pin
        start = current % ring_len
        candidates = ring[start:] + ring[:start]

        # 1) try with hop constraint
        for j in candidates:
            if j == current:
                continue
            hop = abs(j - current); hop = min(hop, n_pins - hop)
            if hop < min_hop:
                continue
            x0, y0 = int(pins[current, 0]), int(pins[current, 1])
            x1, y1 = int(pins[j, 0]), int(pins[j, 1])
            idx = _bresenham_indices(h, w, x0, y0, x1, y1)
            L = idx.size
            if L == 0:
                continue
            score = float(sflat[idx].sum()) / (L + 1e-6)
            if score > best_score:
                best_score, best_j = score, j

        # 2) if nothing found, relax hop and check all
        if best_j is None:
            for j in range(n_pins):
                if j == current:
                    continue
                x0, y0 = int(pins[current, 0]), int(pins[current, 1])
                x1, y1 = int(pins[j, 0]), int(pins[j, 1])
                idx = _bresenham_indices(h, w, x0, y0, x1, y1)
                L = idx.size
                if L == 0:
                    continue
                score = float(sflat[idx].sum()) / (L + 1e-6)
                if score > best_score:
                    best_score, best_j = score, j

        if best_j is None:
            break  # stuck

        # Apply chosen line
        x0, y0 = int(pins[current, 0]), int(pins[current, 1])
        x1, y1 = int(pins[best_j, 0]), int(pins[best_j, 1])
        idx = _bresenham_indices(h, w, x0, y0, x1, y1)
        L = idx.size
        line_mean = float(rflat[idx].sum()) / (L + 1e-6)
        ink = draw_strength * (0.5 + 0.5 * line_mean)

        rflat[idx] -= ink
        np.maximum(residual, 0.0, out=residual)  # clamp

        path.append((current, best_j))
        current = best_j
        error = float(residual.mean())

    if progress_cb:
        progress_cb(100)
    return path, residual, target, pins

def convert_image_to_path(
    img_bgr: np.ndarray,
    n_pins: int,
    steps: int,
    min_hop: int = 6,
    *,
    work_size: int = 512,
    edge_weight: float = 0.5,
    draw_strength: float = 0.10,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> List[Segment]:
    """
    Thin wrapper your worker expects: returns only the path.
    """
    path, _, _, _ = solve_string_art(
        img_bgr=img_bgr,
        n_pins=n_pins,
        steps=steps,
        min_hop=min_hop,
        work_size=work_size,
        edge_weight=edge_weight,
        draw_strength=draw_strength,
        progress_cb=progress_cb,
    )
    return path

def render_path(work_size: int, pins: np.ndarray, path: List[Segment], thickness: int = 1) -> np.ndarray:
    """
    Render the path as white AA lines on black (grayscale uint8); handy for previews.
    """
    canvas = np.zeros((work_size, work_size), dtype=np.uint8)
    for a, b in path:
        x0, y0 = int(pins[a, 0]), int(pins[a, 1])
        x1, y1 = int(pins[b, 0]), int(pins[b, 1])
        cv2.line(canvas, (x0, y0), (x1, y1), 255, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas