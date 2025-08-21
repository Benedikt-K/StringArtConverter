from typing import Callable, List, Optional, Tuple
import math, cv2, numpy as np

def pin_positions_circle(w: int, h: int, n: int, margin: int = 16) -> np.ndarray:
    cx, cy = w // 2, h // 2
    r = max(1, min(w, h) // 2 - margin)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    xs = (cx + r * np.cos(angles)).round().astype(int)
    ys = (cy + r * np.sin(angles)).round().astype(int)
    return np.c_[xs, ys]

def _line_mask(shape: Tuple[int, int], p0: Tuple[int, int], p1: Tuple[int, int], thickness: int = 1) -> np.ndarray:
    """AA line mask in [0,1], single channel float32."""
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.line(m, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
             color=255, thickness=max(1, int(thickness)), lineType=cv2.LINE_AA)
    return (m.astype(np.float32) / 255.0)

def precompute_line_samples(H, W, pins, min_len=0):
    caches = {}
    for i in range(len(pins)):
        x0, y0 = map(int, pins[i])
        for j in range(len(pins)):
            if i == j: continue
            x1, y1 = map(int, pins[j])
            # optional min chord length
            if min_len and np.hypot(x1-x0, y1-y0) < min_len: 
                continue
            length = int(max(1, np.hypot(x1-x0, y1-y0)))
            xs = np.linspace(x0, x1, length).astype(np.int32)
            ys = np.linspace(y0, y1, length).astype(np.int32)
            caches[(i, j)] = (ys, xs)   # index order for row-major images
    return caches