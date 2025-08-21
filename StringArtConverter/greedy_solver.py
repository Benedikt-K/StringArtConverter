from __future__ import annotations
from typing import Callable, List, Optional, Tuple
from .preprocessing import resize, to_grayscale, build_target, remove_background, circular_mask
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
    target_threshold: float = 0.06,
    min_delta: float = 1e-4,
    patience: int = 300,
    min_steps_to_take: int = 700,
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

    # ---------------------- Preprocessing ----------------------
    if (use_backgound_removal):
        img_bgr = remove_background(img_bgr)

    img_bgr = resize(img_bgr, size=work_size)
    gray = to_grayscale(img_bgr, use_clahe=True)
    target = build_target(gray, edge_weight=edge_weight)

    # apply circular mask, so only inside the circle matters
    board_mask = circular_mask(*target.shape, margin=16)
    target *= board_mask

    residual = target.copy()

    # ---------------------- Pins and masks ----------------------
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

    # ---------------------- init candidates ----------------------
    stride = max(1, n_pins // 64)    # ~64 candidates per step
    base = list(range(0, n_pins, stride))
    cand_lists = [(base[i % len(base):] + base[:i % len(base)]) for i in range(n_pins)]
    MAX_CANDS = 64

    # ---------------------- Greedy selection with hop relaxation ----------------------
    path: List[Segment] = []
    current = 0  # always start at pin 0
    min_hop_base = max(int(min_hop), n_pins // 8)
    strength = float(np.clip(draw_strength, 0.01, 1.0))

    recent = [-9999] * n_pins  # for optional mild cooldown
    step_idx = 0

    # plateau initialization
    plateau = 0
    error = residual.mean()

    # multi scale scoring params
    blur_sigma = 0.8               # low-frequency residual
    hi_w = 0.25                    # weight for high-frequency residual
    len_penalty = 0.0005           # penalty for very long lines
    angle_penalty_coeff = 0.03     # discourage repeating the same direction
    prev_vec = None                # last chosen direction (for angle penalty)
    
    # prevent going back to the same pin
    prev_pin = None

    # ---------------------- main loop ----------------------
    while step_idx < steps:
        best_j, best_score = None, -1.0
        hop_req = min_hop_base

        # check if good enough, then stop early, if min steps reached
        if (error < target_threshold) and (step_idx >= min_steps_to_take):
            break

        # try with required hop; if no candidate, relax hop until 0
        while best_j is None and hop_req >= 0:
            for j in range(n_pins):
                if j == current: 
                    continue

                # no immediate backtrack
                if prev_pin is not None and j == prev_pin:
                    continue

                # circular hop distance
                hop = abs(j - current)
                hop = min(hop, n_pins - hop)
                if hop < hop_req: continue

                m = masks[(current, j)]

                """
                # candidate selection for speedup and aviod local "grazing"
                candidates = []
                for j in cand_lists[current]:
                    if j == current:
                        continue
                    hop = abs(j - current); hop = min(hop, n_pins - hop)
                    if hop < hop_req:
                        continue
                    # optional quick angle gate vs. previous vector
                    if prev_vec is not None:
                        v = pins[j] - pins[current]
                        a = prev_vec / (np.linalg.norm(prev_vec) + 1e-9)
                        b = v / (np.linalg.norm(v) + 1e-9)
                        if float(np.dot(a, b)) > 0.97:  # nearly parallel; skip
                            continue
                    candidates.append(j)
                    if len(candidates) >= MAX_CANDS:
                        break

                if not candidates:  # fallback
                    candidates = [j for j in range(n_pins) if j != current]
                """

                # multi-scale score
                score_multi = cv2.GaussianBlur(residual, (0, 0), blur_sigma)
                hi = residual - score_multi
                score_val = ((0.8 * score_multi + hi_w * hi) * m).sum()

                # normalize and length penalty
                score_val /= (lens[(current, j)] + 1e-6) 
                score_val /= (1.0 + len_penalty * lens[(current, j)])

                # angle penalty
                if prev_vec is not None:
                    v = pins[j] - pins[current]
                    a = prev_vec / (np.linalg.norm(prev_vec) + 1e-9)
                    b = v / (np.linalg.norm(v) + 1e-9)
                    cos_sim = float(np.clip(np.dot(a, b), -1.0, 1.0))  # 1 = same direction
                    score_val -= angle_penalty_coeff * cos_sim

                # light cooldown: discourage bouncing to very recent pin
                if (step_idx - recent[j]) < 10:
                    score_val -= 0.02

                if score_val > best_score:
                    best_score, best_j = score_val, j

            if best_j is None:
                hop_req -= 1  # relax and retry

        if best_j is None:
            # truly stuck (should be rare) break to avoid infinite loop
            break

        # apply chosen line
        m = masks[(current, best_j)]
        line_mean = float((residual * m).sum() / lens[(current, best_j)] + 1e-6)

        # adaptive "ink"
        ink = strength * (0.5 + 0.5 * line_mean)
        residual -= ink * m   
        np.maximum(residual, 0.0, out=residual)

        # track for next step 
        prev_vec = pins[best_j] - pins[current]
        prev_pin = current

        # update error for early stopping
        prev_error = error
        error = residual.mean()
        improvement = prev_error - error
        relative_improvement = improvement / (prev_error + 1e-9)

        if (improvement < min_delta) and (relative_improvement < 0.002):    # 0.2% relative improvement here
            plateau += 1
        else:
            plateau = 0

        # no meaningful improvement over time
        if (plateau >= patience) and (step_idx >= min_steps_to_take):
            break

        # commit step to list
        path.append((current, best_j))
        recent[best_j] = step_idx
        current = best_j
        step_idx += 1

        if progress_cb:
            #progress_cb(int(100 * step_idx / max(1, steps)))
            progress_cb(int(100 * (1.0 - min(1.0, error))))

    if progress_cb:
        progress_cb(100)
    return path