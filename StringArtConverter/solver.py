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