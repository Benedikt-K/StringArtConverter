# main.py  (artifact-resistant greedy solver with full CLI knobs)
from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import argparse, math
import cv2, numpy as np

Segment = Tuple[int, int]

# -------------------- Preprocessing --------------------

def resize_square(img_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

def to_grayscale(img_bgr: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
    return g

def build_target(gray_u8: np.ndarray, edge_weight: float = 0.7) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0
    darkness = 1.0 - gray
    med = float(np.median(gray_u8))
    t1 = int(max(0, 0.66 * med))
    t2 = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray_u8, t1, t2).astype(np.float32) / 255.0
    tgt = (1.0 - edge_weight) * darkness + edge_weight * edges
    return np.clip(tgt, 0.0, 1.0)

def circular_mask(h: int, w: int, margin: int = 16) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w * 0.5, h * 0.5
    r = min(h, w) * 0.5 - margin
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.float32)

# -------------------- Geometry --------------------

def pin_positions_circle(h: int, w: int, n_pins: int, margin: int = 16) -> np.ndarray:
    cx, cy = w / 2.0, h / 2.0
    r = min(h, w) / 2.0 - margin
    ang = np.linspace(0.0, 2.0 * math.pi, n_pins, endpoint=False)
    xs = (cx + r * np.cos(ang)).round().astype(np.int32)
    ys = (cy + r * np.sin(ang)).round().astype(np.int32)
    return np.stack([xs, ys], axis=1)

def bresenham_indices(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    dx = abs(x1 - x0); dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy; x, y = x0, y0
    xs, ys = [], []
    while True:
        if 0 <= x < w and 0 <= y < h:
            xs.append(x); ys.append(y)
        if x == x1 and y == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x += sx
        if e2 <= dx: err += dx; y += sy
    if not xs: return np.empty(0, np.int32)
    return (np.asarray(ys, np.int32) * w + np.asarray(xs, np.int32))

# -------------------- Candidate generation --------------------

def candidate_pool(
    current: int,
    n_pins: int,
    budget: int,
    min_hop: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Blend of 'balanced' (symmetric) and random candidates.
    ~60% balanced, ~40% random; both respect min_hop.
    """
    need = max(8, budget)
    half_bal = max(2, int(0.6 * need) // 2)
    step = max(1, n_pins // max(2, half_bal))

    out: List[int] = []
    seen = set([current])

    # Balanced around current
    for m in range(1, half_bal + 1):
        for s in (+1, -1):
            j = (current + s * m * step) % n_pins
            if j in seen: continue
            hop = abs(j - current); hop = min(hop, n_pins - hop)
            if hop >= min_hop:
                out.append(j); seen.add(j)

    # Random fill
    while len(out) < need and len(seen) < n_pins:
        j = int(rng.integers(0, n_pins))
        if j in seen: continue
        hop = abs(j - current); hop = min(hop, n_pins - hop)
        if hop >= min_hop:
            out.append(j); seen.add(j)

    return out

# -------------------- Solver (with angle penalty, cooldown, center bonus) --------------------

def solve_string_art(
    img_bgr: np.ndarray,
    n_pins: int,
    steps: int,
    *,
    work_size: int = 640,
    min_hop: int = 3,
    edge_weight: float = 0.78,
    draw_strength: float = 0.08,         # residual subtraction per line (adaptive)
    candidate_budget: int = 128,         # candidates tested per step
    blur_every: int = 40,                # mild diffusion every N steps
    angle_penalty: float = 0.12,         # penalize repeating the same direction
    cooldown_steps: int = 18,            # discourage revisiting a pin too soon
    cooldown_penalty: float = 0.15,      # strength of cooldown penalty
    center_rel_radius: float = 0.10,     # FRACTION of min(H,W) considered "near center"
    center_bonus: float = 0.10,          # FRACTION of cooldown_penalty added if near center
    random_seed: int = 42,               # deterministic runs
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Tuple[List[Segment], np.ndarray, np.ndarray, np.ndarray]:

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image")
    if n_pins < 3:
        raise ValueError("n_pins must be >= 3")

    rng = np.random.default_rng(random_seed)

    img_bgr = resize_square(img_bgr, work_size)
    gray = to_grayscale(img_bgr, use_clahe=True)
    target = build_target(gray, edge_weight=edge_weight)
    mask = circular_mask(*target.shape, margin=16)
    target *= mask

    residual = target.astype(np.float32).copy()
    H, W = residual.shape
    rflat = residual.ravel()

    pins = pin_positions_circle(H, W, n_pins, margin=16).astype(np.int32)
    path: List[Segment] = []
    current = 0
    draw_strength = float(np.clip(draw_strength, 0.01, 0.5))

    # Track recent usage for cooldown & previous direction for angle penalty
    last_used = np.full(n_pins, -10_000, dtype=np.int32)
    prev_vec = None
    prev_pin = None

    # center parameters (convert from fractions)
    cx, cy = W * 0.5, H * 0.5
    center_radius_px = float(center_rel_radius) * min(H, W)
    center_bonus_abs = float(center_bonus) * float(cooldown_penalty)

    for step in range(steps):
        if progress_cb:
            progress_cb(int(100 * step / max(1, steps - 1)))

        if blur_every > 0 and step % blur_every == 0 and step > 0:
            cv2.GaussianBlur(residual, (0, 0), 0.8, dst=residual)

        best_j, best_score = None, -1e9
        cands = candidate_pool(current, n_pins, candidate_budget, min_hop, rng)

        for j in cands:
            if j == current or (prev_pin is not None and j == prev_pin):
                continue

            x0, y0 = int(pins[current, 0]), int(pins[current, 1])
            x1, y1 = int(pins[j, 0]), int(pins[j, 1])
            idx = bresenham_indices(H, W, x0, y0, x1, y1)
            L = idx.size
            if L == 0:
                continue

            # Base score = average residual along the line
            score = float(rflat[idx].sum()) / (L + 1e-6)

            # Angle penalty (only for near-parallel moves, not cross-center)
            if prev_vec is not None:
                v = np.array([x1 - x0, y1 - y0], dtype=np.float32)
                a = prev_vec / (np.linalg.norm(prev_vec) + 1e-9)
                b = v / (np.linalg.norm(v) + 1e-9)
                cos_sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
                if cos_sim > 0.85:
                    score -= angle_penalty * cos_sim

            # Cooldown: reduce score for pins touched very recently
            age = step - last_used[j]
            if age < cooldown_steps:
                score -= cooldown_penalty * (1.0 - age / max(1, cooldown_steps))

            # Center-aware cooldown relaxation: if line passes near center, add bonus
            mid_x, mid_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            if (mid_x - cx) ** 2 + (mid_y - cy) ** 2 < center_radius_px ** 2:
                score += center_bonus_abs

            if score >= best_score:
                best_score, best_j = score, j

        # Fallback: if no candidate passed filters, relax & try all pins
        if best_j is None:
            for j in range(n_pins):
                if j == current:
                    continue
                x0, y0 = int(pins[current, 0]), int(pins[current, 1])
                x1, y1 = int(pins[j, 0]), int(pins[j, 1])
                idx = bresenham_indices(H, W, x0, y0, x1, y1)
                L = idx.size
                if L == 0:
                    continue
                score = float(rflat[idx].sum()) / (L + 1e-6)
                if score > best_score:
                    best_score, best_j = score, j

        if best_j is None:
            break  # stuck

        # Apply chosen line (adaptive "ink")
        x0, y0 = int(pins[current, 0]), int(pins[current, 1])
        x1, y1 = int(pins[best_j, 0]), int(pins[best_j, 1])
        idx = bresenham_indices(H, W, x0, y0, x1, y1)
        L = idx.size
        line_mean = float(rflat[idx].sum()) / (L + 1e-6)
        ink = draw_strength * (0.5 + 0.5 * line_mean)

        rflat[idx] -= ink
        np.maximum(residual, 0.0, out=residual)

        path.append((current, best_j))
        last_used[current] = step
        prev_vec = np.array([x1 - x0, y1 - y0], dtype=np.float32)
        prev_pin = current
        current = best_j

    if progress_cb:
        progress_cb(100)
    return path, residual, target, pins

# -------------------- Preview & Path I/O --------------------

def render_path(
    work_size: int,
    pins: np.ndarray,
    path: List[tuple[int, int]],
    thickness: int = 1,
    alpha_per_line: float = 0.12,
    circle_board: bool = True,
    margin: int = 16,
    gamma: float = 1.0,
) -> np.ndarray:
    H = W = work_size
    acc = np.zeros((H, W), dtype=np.float32)
    if circle_board:
        yy, xx = np.ogrid[:H, :W]
        cx, cy = W * 0.5, H * 0.5
        r = min(H, W) * 0.5 - margin
        board = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    else:
        board = np.ones((H, W), dtype=bool)

    for a, b in path:
        tmp = np.zeros_like(acc)
        x0, y0 = int(pins[a, 0]), int(pins[a, 1])
        x1, y1 = int(pins[b, 0]), int(pins[b, 1])
        cv2.line(tmp, (x0, y0), (x1, y1), 1.0, thickness=thickness, lineType=cv2.LINE_AA)
        acc[board] += alpha_per_line * tmp[board]

    acc = np.clip(acc, 0.0, 1.0)
    if gamma != 1.0:
        acc = acc ** (1.0 / gamma)
    return (255.0 * (1.0 - acc)).astype(np.uint8)

def save_path_txt(path: List[Segment], out_txt: str) -> None:
    with open(out_txt, "w", encoding="utf-8") as f:
        for a, b in path:
            f.write(f"{a} {b}\n")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="String-art converter (artifact-resistant greedy).")
    # Core
    ap.add_argument("--input", required=True)
    ap.add_argument("--pins", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--work_size", type=int, default=512)
    ap.add_argument("--min_hop", type=int, default=6)
    ap.add_argument("--edge_weight", type=float, default=0.55)
    ap.add_argument("--draw_strength", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)

    # Solver knobs
    ap.add_argument("--candidate_budget", type=int, default=128)
    ap.add_argument("--blur_every", type=int, default=40)
    ap.add_argument("--angle_penalty", type=float, default=0.12)
    ap.add_argument("--cooldown_steps", type=int, default=18)
    ap.add_argument("--cooldown_penalty", type=float, default=0.15)
    ap.add_argument("--center_rel_radius", type=float, default=0.10, help="fraction of min(H,W)")
    ap.add_argument("--center_bonus", type=float, default=0.10, help="fraction of cooldown_penalty")

    # Preview
    ap.add_argument("--preview", default="")
    ap.add_argument("--out", default="path.txt")
    ap.add_argument("--render_alpha", type=float, default=0.10)
    ap.add_argument("--render_gamma", type=float, default=1.2)
    ap.add_argument("--line_thickness", type=int, default=1)

    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read: {args.input}")

    def progress(p: int):
        print(f"\rProgress: {p:3d}%", end="", flush=True)

    path, residual, target, pins = solve_string_art(
        img_bgr=img,
        n_pins=args.pins,
        steps=args.steps,
        work_size=args.work_size,
        min_hop=args.min_hop,
        edge_weight=args.edge_weight,
        draw_strength=args.draw_strength,
        candidate_budget=args.candidate_budget,
        blur_every=args.blur_every,
        angle_penalty=args.angle_penalty,
        cooldown_steps=args.cooldown_steps,
        cooldown_penalty=args.cooldown_penalty,
        center_rel_radius=args.center_rel_radius,
        center_bonus=args.center_bonus,
        random_seed=args.seed,
        progress_cb=progress,
    )
    print("\nDone.")
    save_path_txt(path, args.out)
    print(f"Saved {len(path)} segments to {args.out}")

    if args.preview:
        preview = render_path(
            args.work_size, pins, path,
            thickness=args.line_thickness,
            alpha_per_line=args.render_alpha,
            gamma=args.render_gamma
        )
        cv2.imwrite(args.preview, preview)
        print(f"Saved preview to {args.preview}")

if __name__ == "__main__":
    main()
