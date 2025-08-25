# go_main.py  (Go-algorithm faithful port, with CLI + preview)
from __future__ import annotations
from typing import List, Tuple, Optional
import argparse, math
import cv2, numpy as np

Segment = Tuple[int, int]

# ---------- Preprocessing (modular) ----------

def resize_square(img_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

def to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def apply_clahe(gray_u8: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    return clahe.apply(gray_u8)

def contrast_stretch(gray_u8: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    # percentile-based min/max → stretch to [0,255]
    lo = np.percentile(gray_u8, np.clip(p_low, 0, 50))
    hi = np.percentile(gray_u8, np.clip(p_high, 50, 100))
    if hi <= lo + 1e-6:
        return gray_u8.copy()
    g = np.clip((gray_u8.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (g * 255.0 + 0.5).astype(np.uint8)

def canny_edges(gray_u8: np.ndarray, low: int = -1, high: int = -1, auto_sigma: float = 0.33) -> np.ndarray:
    if low < 0 or high < 0:
        # auto thresholds from median
        v = np.median(gray_u8)
        low = int(max(0, (1.0 - auto_sigma) * v))
        high = int(min(255, (1.0 + auto_sigma) * v))
    return cv2.Canny(gray_u8, low, high)

def build_brightness_for_go_solver(
    img_bgr: np.ndarray,
    *,
    work_size: int,
    use_clahe: bool,
    use_contrast: bool,
    p_low: float,
    p_high: float,
    use_edges: bool,
    edge_weight: float,
    edge_low: int,
    edge_high: int,
    edge_auto_sigma: float,
) -> np.ndarray:
    """
    Returns uint8 brightness image (H,W) where 0=black, 255=white.
    Internally we form a 'target darkness' in [0..1], then convert:
      SourceImage_u8 = 255 * (1 - target_darkness)
    so dark/edgey areas become low brightness → high error (255 - src).
    """
    img = resize_square(img_bgr, work_size)
    gray = to_gray_u8(img)

    if use_clahe:
        gray = apply_clahe(gray)

    if use_contrast:
        gray = contrast_stretch(gray, p_low=p_low, p_high=p_high)

    gray_f = gray.astype(np.float32) / 255.0
    darkness = 1.0 - gray_f  # 1 = needs thread

    if use_edges:
        e = canny_edges(gray, low=edge_low, high=edge_high, auto_sigma=edge_auto_sigma).astype(np.float32) / 255.0
        # blend: more weight → more emphasis on contours
        target_dark = (1.0 - edge_weight) * darkness + edge_weight * e
    else:
        target_dark = darkness

    # convert back to Go-style brightness
    src = (255.0 * (1.0 - target_dark)).astype(np.uint8)
    return src

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

# -------------------- Rendering --------------------

def render_path(work_size, pins, path, alpha_per_line=0.10,
                gamma=1.0, thickness=1, margin=16):
    """Render preview image from path with tunable darkness and gamma."""
    H = W = work_size
    acc = np.zeros((H, W), dtype=np.float32)

    # circular board mask
    yy, xx = np.ogrid[:H, :W]
    cx, cy = W * 0.5, H * 0.5
    r = min(H, W) * 0.5 - margin
    board = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r

    for a, b in path:
        tmp = np.zeros_like(acc)
        x0, y0 = int(pins[a][0]), int(pins[a][1])
        x1, y1 = int(pins[b][0]), int(pins[b][1])
        cv2.line(tmp, (x0, y0), (x1, y1), 1.0,
                 thickness=thickness, lineType=cv2.LINE_AA)
        acc[board] += alpha_per_line * tmp[board]

    acc = np.clip(acc, 0.0, 1.0)
    if gamma != 1.0:
        acc = acc ** (1.0 / gamma)

    img = (255.0 * (1.0 - acc)).astype(np.uint8)
    return img

def save_path_txt(path: List[Segment], out_txt: str) -> None:
    with open(out_txt, "w", encoding="utf-8") as f:
        for a, b in path:
            f.write(f"{a} {b}\n")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="String-art converter (faithful Go port).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--pins", type=int, default=300)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--work_size", type=int, default=500)
    ap.add_argument("--min_distance", type=int, default=30)
    ap.add_argument("--line_weight", type=float, default=8.0)
    ap.add_argument("--last_n", type=int, default=20)
    ap.add_argument("--preview", default="")
    ap.add_argument("--out", default="path.txt")
    ap.add_argument("--render_alpha", type=float, default=0.10, help="Per-line darkness contribution in preview (smaller = lighter)")
    ap.add_argument("--render_gamma", type=float, default=1.0, help="Gamma correction for preview (1.0 = none)")
    ap.add_argument("--line_thickness", type=int, default=1, help="Thickness of preview lines")
    # --- Preprocessing knobs ---
    ap.add_argument("--pp_clahe", action="store_true", help="Apply CLAHE before other steps")
    ap.add_argument("--pp_contrast", action="store_true", help="Percentile contrast stretch")
    ap.add_argument("--pp_c_low", type=float, default=2.0, help="Contrast low percentile (0..50)")
    ap.add_argument("--pp_c_high", type=float, default=98.0, help="Contrast high percentile (50..100)")

    ap.add_argument("--pp_edges", action="store_true", help="Add Canny edges to the target (blended)")
    ap.add_argument("--pp_edge_weight", type=float, default=0.35, help="Blend weight for edges (0..1)")
    ap.add_argument("--pp_edge_low", type=int, default=-1, help="Canny low threshold; -1 = auto")
    ap.add_argument("--pp_edge_high", type=int, default=-1, help="Canny high threshold; -1 = auto")
    ap.add_argument("--pp_edge_auto_sigma", type=float, default=0.33, help="Auto Canny sigma (median-based)")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read: {args.input}")

    def progress(p: int):
        print(f"\rProgress: {p:3d}%", end="", flush=True)

    # preprocessing
    src_u8 = build_brightness_for_go_solver(
        img_bgr=img,
        work_size=args.work_size,
        use_clahe=args.pp_clahe,
        use_contrast=args.pp_contrast,
        p_low=args.pp_c_low,
        p_high=args.pp_c_high,
        use_edges=args.pp_edges,
        edge_weight=args.pp_edge_weight,
        edge_low=args.pp_edge_low,
        edge_high=args.pp_edge_high,
        edge_auto_sigma=args.pp_edge_auto_sigma,
    )
    # Flatten to match the Go logic (row-major)
    H = W = args.work_size
    SourceImg = src_u8.reshape(H * W).astype(np.float64)

    # find path
    path, error, target, pins = solve_string_art_go(
        source_brightness_u8=src_u8,
        n_pins=args.pins,
        max_lines=args.steps,
        min_distance=args.min_distance,
        line_weight=args.line_weight,
        last_n=args.last_n,
        work_size=args.work_size,
        progress_cb=progress,
    )
    print("\nDone.")

    save_path_txt(path, args.out)
    print(f"Saved {len(path)} segments to {args.out}")

    if args.preview:
        preview = render_path(
            args.work_size, pins, path,
            alpha_per_line=args.render_alpha,
            gamma=args.render_gamma,
            thickness=args.line_thickness
        )
        cv2.imwrite(args.preview, preview)
        print(f"Saved preview to {args.preview}")

if __name__ == "__main__":
    main()
