from __future__ import annotations
import argparse
import cv2, numpy as np

from StringArtConverter.preprocessing import build_target_for_solver
from StringArtConverter.previewer import render_path
from StringArtConverter.solver import solve_string_art
from StringArtConverter.utils import save_path_txt

def main():
    """
    Run the command-line interface for the String Art Converter.

    This function parses command-line arguments, applies preprocessing
    to the input image, runs the string-art solver, and optionally renders
    a preview image.

    The typical workflow is:
        1. Load the input image from disk.
        2. Preprocess it according to the given options (CLAHE, contrast, edges, etc.).
        3. Run the string-art solver to compute an optimal path between nails.
        4. Save the path to a text file.
        5. Optionally, render a preview image showing the generated thread pattern.

    Command-line arguments
    ----------------------
    Required:
        --input : str
            Path to the input image file.

    Solver parameters:
        --pins : int
            Number of nails/pins around the frame.
        --steps : int
            Maximum number of lines to draw.
        --work_size : int
            Image size used internally for computation.
        --min_distance : int
            Minimum distance (in pins) between consecutive connections.
        --line_weight : float
            Darkness contribution per line (for simulation).
        --last_n : int
            Number of recent lines ignored to avoid overlap.
        --preview : str
            Optional output filename for the rendered preview image.
        --out : str
            Path to save the computed path text file.

    Rendering parameters:
        --render_alpha : float
            Per-line darkness contribution in the preview (smaller = lighter).
        --render_gamma : float
            Gamma correction applied to the rendered preview.
        --line_thickness : int
            Thickness of lines in the rendered preview image.

    Preprocessing parameters:
        --pp_clahe : bool
            Apply CLAHE before other steps.
        --pp_contrast : bool
            Enable percentile-based contrast stretching.
        --pp_c_low / --pp_c_high : float
            Percentiles for low/high contrast adjustment.
        --pp_rembg : bool
            Use rembg to separate foreground and dim the background.
        --pp_rembg_dim : float
            Factor for dimming background intensity (0-1).
        --pp_rembg_feather : int
            Feathering (Gaussian blur) for mask edges.
        --pp_rembg_erode : int
            Pixel erosion amount to reduce halo artifacts.
        --pp_gamma : float
            Gamma correction before inversion (<1 = brighten, >1 = darken).
        --pp_clip_high : float
            Percentile for brightness clipping (e.g., 95 = clip top 5%).
        --pp_edges : bool
            Blend Canny edges into the target image.
        --pp_edge_weight : float
            Weight for blending edges (0-1).
        --pp_edge_low / --pp_edge_high : int
            Canny edge thresholds; -1 = auto.
        --pp_edge_auto_sigma : float
            Sigma factor used for automatic Canny thresholds.

    Raises:
        SystemExit: If the input image cannot be loaded.

    Side Effects:
        - Prints progress to stdout.
        - Writes output files (path text file, optional preview image).

    Example:
        $ python CLI.py --input portrait.jpg --pins 300 --steps 3000 --preview output.png
    """
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
    src_u8 = build_target_for_solver(
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
    path, error, target, pins = solve_string_art(
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
