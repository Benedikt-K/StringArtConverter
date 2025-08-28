from __future__ import annotations
import argparse
import cv2, numpy as np
from typing import Tuple, List

from StringArtConverter.preprocessing import build_brightness_for_solver
from StringArtConverter.previewer import render_path
from StringArtConverter.solver import solve_string_art_go
from StringArtConverter.utils import save_path_txt

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="String-art converter")
    # --- solver ---
    ap.add_argument("--input", required=True)
    ap.add_argument("--pins", type=int, default=300)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--work_size", type=int, default=500)
    ap.add_argument("--min_distance", type=int, default=30)
    ap.add_argument("--line_weight", type=float, default=8.0)
    ap.add_argument("--last_n", type=int, default=20)
    ap.add_argument("--preview", default="")
    ap.add_argument("--out", default="path.txt")
    # --- Rendering ---
    ap.add_argument("--render_alpha", type=float, default=0.10, help="Per-line darkness contribution in preview (smaller = lighter)")
    ap.add_argument("--render_gamma", type=float, default=1.0, help="Gamma correction for preview (1.0 = none)")
    ap.add_argument("--line_thickness", type=int, default=1, help="Thickness of preview lines")
    # --- Preprocessing ---
    ap.add_argument("--pp_clahe", action="store_true", help="Apply CLAHE before other steps")
    ap.add_argument("--pp_contrast", action="store_true", help="Percentile contrast stretch")
    ap.add_argument("--pp_c_low", type=float, default=2.0, help="Contrast low percentile (0..50)")
    ap.add_argument("--pp_c_high", type=float, default=98.0, help="Contrast high percentile (50..100)")
    ap.add_argument("--pp_rembg", action="store_true", help="Use rembg to get a foreground mask and darken background")
    ap.add_argument("--pp_rembg_dim", type=float, default=0.45, help="How much to darken the background (0..1)")
    ap.add_argument("--pp_rembg_feather", type=int, default=6, help="Feather (Gaussian sigma in px) for bg mask edges")
    ap.add_argument("--pp_rembg_erode", type=int, default=1, help="Erode foreground mask in px to reduce hair halos")
    ap.add_argument("--pp_gamma", type=float, default=1.0, help="Gamma correction before inversion ( <1 = brighten, >1 = darken highlights )")
    ap.add_argument("--pp_clip_high", type=float, default=100.0, help="Percentile high clipping for brightness (e.g. 95 = ignore brightest 5%)")
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
    src_u8 = build_brightness_for_solver(
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
        use_rembg=args.pp_rembg,
        rembg_dim=args.pp_rembg_dim,
        rembg_feather=args.pp_rembg_feather,
        rembg_erode=args.pp_rembg_erode,
        pp_gamma=args.pp_gamma,
        pp_clip_high=args.pp_clip_high,
    )
    # Flatten
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
