from __future__ import annotations
from typing import Callable, List, Optional, Tuple
from .preprocessing import resize, to_grayscale, build_target, remove_background
import math, cv2, numpy as np

Coord = Tuple[int, int]
Segment = Tuple[int, int]

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


def convert_image_to_path(
    img_bgr: np.ndarray,
    n_pins: int,
    steps: int,
    min_hop: int = 6,
    work_size: int = 512,
    edge_weight: float = 0.65,
    draw_strength: float = 0.10,
    progress_cb: Optional[Callable[[int], None]] = None,
    use_backgound_removal=False,
    traget_threshold: float = 0.06,
    min_delta: float = 1e-4,
    patience: int = 200,
    min_steps_to_take: int = 500,
) -> List[Segment]:
    """
    Improved greedy solver:
      - works on a square canvas (work_size x work_size)
      - target = 0.6*dark + 0.4*edges (tunable via edge_weight)
      - per-line AA masks; length-normalized scores; residual clamped to [0,1]
      - relaxes hop if no candidates so it reaches 'steps'
    Returns a list of (from_pin, to_pin) pin indices.
    """
    print("Converting...")
    if img_bgr is None or img_bgr.size == 0:
        return []

    # Preprocessing
    if (use_backgound_removal):
        img_bgr = remove_background(img_bgr)

    img_bgr = resize(img_bgr, work_size=work_size)
    gray = to_grayscale(img_bgr, use_clahe=True)
    target = build_target(gray, edge_weight=edge_weight)

    residual = target.copy()

    # Pins and precomputed masks
    pins = pin_positions_circle(work_size, work_size, n_pins, margin=16)
    H, W = residual.shape
    masks: dict[Tuple[int, int], np.ndarray] = {}
    lens: dict[Tuple[int, int], float] = {}

    for i in range(n_pins):
        for j in range(n_pins):
            if i == j: continue
            p0, p1 = pins[i], pins[j]
            m = _line_mask((H, W), (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), thickness=1)
            s = float(m.sum()) + 1e-6
            masks[(i, j)] = m
            lens[(i, j)] = s

    # Greedy selection with hop relaxation
    path: List[Segment] = []
    current = 0  # start somewhere deterministic
    #min_hop_base = max(0, int(min_hop))
    min_hop_base = max(int(min_hop), n_pins // 8)
    strength = float(np.clip(draw_strength, 0.01, 1.0))

    recent = [-9999] * n_pins  # for optional mild cooldown
    step_idx = 0

    # plateau initialization
    plateau = 0
    error = residual.mean()

    while step_idx < steps:
        best_j, best_score = None, -1.0
        hop_req = min_hop_base

        # check if close enough, then stop early, if min steps reached
        if error < traget_threshold and step_idx >= min_steps_to_take :
            break

        # try with required hop; if no candidate, relax hop until 0
        while best_j is None and hop_req >= 0:
            for j in range(n_pins):
                if j == current: continue
                # circular hop distance
                hop = abs(j - current)
                hop = min(hop, n_pins - hop)
                if hop < hop_req: continue

                m = masks[(current, j)]
                s = float((residual * m).sum()) / lens[(current, j)]  # length-normalized
                # light cooldown: discourage bouncing to very recent pin
                cooldown_penalty = 0.02 if (step_idx - recent[j]) < 10 else 0.0
                s -= cooldown_penalty

                if s > best_score:
                    best_score, best_j = s, j

            if best_j is None:
                hop_req -= 1  # relax and retry

        if best_j is None:
            # truly stuck (should be rare); break to avoid infinite loop
            break

        # update residual where the line is
        m = masks[(current, best_j)]
        residual -= strength * m**0.9   # added 0.9 here TODO figure out if nescessary
        np.maximum(residual, 0.0, out=residual)  # clamp to [0,1]

        # update error
        prev_error = error
        error = residual.mean()
        improvement = prev_error - error
        relative_improvement = improvement / (prev_error + 1e-9)

        if improvement < min_delta and relative_improvement < 0.002:    # 0.2% relative improvement here
            plateau += 1
        else:
            plateau = 0

        # no meaningful improvement over time
        if plateau >= patience:
            break

        path.append((current, best_j))
        recent[best_j] = step_idx
        current = best_j
        step_idx += 1

        if progress_cb:
            progress_cb(int(100 * step_idx / max(1, steps)))

    if progress_cb:
        progress_cb(100)
    return path