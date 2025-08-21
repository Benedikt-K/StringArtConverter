from typing import List, Tuple, Optional, Callable
from .solver import pin_positions_circle, _line_mask
from .preprocessing import remove_background
from .preprocessing import resize, to_grayscale, build_target, circular_mask
import numpy as np
import cv2

Segment = Tuple[int, int]

def solve_multi_strings(
    img_bgr: np.ndarray,
    n_pins: int,
    steps: int,
    *,
    k_strings: int = 3,               # how many strings
    start_pins: Optional[List[int]] = None,
    min_hop: int = 6,
    work_size: int = 512,
    edge_weight: float = 0.55,
    tone_weight: float = 0.20,
    draw_strength: float = 0.10,
    progress_cb: Optional[Callable[[int], None]] = None,
    use_background_removal: bool = False,
) -> Tuple[List[Segment], List[List[Segment]]]:
    """
    Multi-string greedy: K independent cursors take turns.
    All update the SAME residual, so they collaborate.
    """
    if img_bgr is None or img_bgr.size == 0:
        return [], [[] for _ in range(k_strings)]

    # --- preprocessing ---
    if use_background_removal:
        img_bgr = remove_background(img_bgr)


    img_bgr = resize(img_bgr, work_size, mode="cover")
    gray = to_grayscale(img_bgr, use_clahe=True)
    target = build_target(gray, edge_weight=edge_weight, tone_weight=tone_weight)

    H, W = target.shape
    board_mask = circular_mask(H, W, margin=16)
    target *= board_mask
    residual = target.copy()

    # coverage penalty
    coverage = np.zeros_like(residual, np.float32)
    cov_w = 0.12  # range 0.08–0.18

    # --- pins & masks ---
    pins = pin_positions_circle(work_size, work_size, n_pins, margin=16)
    masks, lens = {}, {}
    for i in range(n_pins):
        for j in range(n_pins):
            if i == j: continue
            p0, p1 = pins[i], pins[j]
            m = _line_mask((H, W), (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), thickness=1)
            s = float(m.sum()) + 1e-6
            masks[(i, j)] = m
            lens[(i, j)] = s

    # --- state for K strings ---
    if not start_pins:
        # evenly spaced starts
        start_pins = [int(i * n_pins / k_strings) % n_pins for i in range(k_strings)]
    current = list(start_pins)
    prev_pin = [None] * k_strings
    prev_vec = [None] * k_strings
    recent = [[-9999] * n_pins for _ in range(k_strings)]

    min_hop_base = max(int(min_hop), n_pins // 8)
    strength = float(np.clip(draw_strength, 0.01, 1.0))

    blur_sigma = 0.8
    hi_w = 0.30
    len_penalty = 0.0007
    angle_penalty = 0.03

    paths_per_string: List[List[Segment]] = [[] for _ in range(k_strings)]
    combined_path: List[Segment] = []

    total_steps = 0
    while total_steps < steps:
        s_idx = total_steps % k_strings  # round-robin which string moves now
        cur = current[s_idx]

        best_j, best_score = None, -1.0
        hop_req = min_hop_base

        # try with required hop; if none, relax
        while best_j is None and hop_req >= 0:
            for j in range(n_pins):
                if j == cur: continue

                # no immediate backtrack for this string
                if prev_pin[s_idx] is not None and j == prev_pin[s_idx]:
                    continue

                hop = abs(j - cur); hop = min(hop, n_pins - hop)
                if hop < hop_req: continue

                m = masks[(cur, j)]
                # multi-scale score
                sm = cv2.GaussianBlur(residual, (0, 0), blur_sigma)
                hi = residual - sm
                score_val = ((0.8 * sm + hi_w * hi) * m).sum()
                score_val /= (lens[(cur, j)] + 1e-6)
                score_val /= (1.0 + len_penalty * lens[(cur, j)])

                # coverage penalty
                score_val -= cov_w * float((coverage * m).sum()) / (lens[(cur, j)] + 1e-6)

                if prev_vec[s_idx] is not None:
                    v = pins[j] - pins[cur]
                    a = prev_vec[s_idx] / (np.linalg.norm(prev_vec[s_idx]) + 1e-9)
                    b = v / (np.linalg.norm(v) + 1e-9)
                    score_val -= angle_penalty * float(np.clip(np.dot(a, b), -1.0, 1.0))

                # light per-string cooldown
                if (total_steps - recent[s_idx][j]) < 10:
                    score_val -= 0.02

                if score_val > best_score:
                    best_score, best_j = score_val, j

            if best_j is None:
                hop_req -= 1

        if best_j is None:
            # this string can’t move; try next string
            total_steps += 1
            if progress_cb:
                progress_cb(int(100 * total_steps / max(1, steps)))
            continue

        # adaptive ink based on current residual along this line
        m = masks[(cur, best_j)]
        line_mean = float((residual * m).sum()) / (lens[(cur, best_j)] + 1e-6)
        ink = strength * (0.5 + 0.5 * line_mean)
        residual -= ink * m
        np.maximum(residual, 0.0, out=residual)

        # record
        paths_per_string[s_idx].append((cur, best_j))
        combined_path.append((cur, best_j))
        recent[s_idx][best_j] = total_steps
        prev_vec[s_idx] = pins[best_j] - pins[cur]
        prev_pin[s_idx] = cur
        current[s_idx] = best_j
        coverage += m

        total_steps += 1
        if progress_cb:
            progress_cb(int(100 * total_steps / max(1, steps)))

    if progress_cb:
        progress_cb(100)
    return combined_path, paths_per_string