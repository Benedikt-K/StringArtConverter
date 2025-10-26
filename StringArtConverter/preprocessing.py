"""
preprocessing.py

Image preprocessing utilities for the String Art Converter.

This module provides a set of helper functions for preparing images
before they are processed by the string art solver. 
"""

from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp

try:
    from rembg import remove as rembg_remove
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False

def resize_square(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """
    Center crop an image, then resize it to the desired size.

    Args:
        img_gbr (np.ndarray): Input image in BGR format
        size (int): Target side length in pixels

    Returns:
        np.darray: Square image of shape (size, size, 3)
    """
    h, w = img_bgr.shape[:2]
    side = min(h, w)

    y0 = (h - side) // 2
    x0 = (w - side) // 2

    square = img_bgr[y0:y0 + side, x0:x0 + side]

    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

def to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale (uint8).

    Args:
        img_bgr (np.ndarray): Input color image in BGR format.

    Returns:
        np.ndarray: Grayscale image
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def apply_clahe(gray_u8: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to image.

    Args:
        gray_u8 (np.ndarray): Grayscale input image.
        clip (float, optional): CLAHE clip limit. Defaults to 2.0.
        tiles (int, optional): Tile grid size (tiles x tiles). Defaults to 8.

    Returns:
        np.ndarray: Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    return clahe.apply(gray_u8)

def contrast_stretch(gray_u8: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """
    Perform percentile-based contrast stretching.

    Args:
        gray_u8 (np.ndarray): Grayscale input image.
        p_low (float, optional): Low percentile cutoff. Defaults to 2.0.
        p_high (float, optional): High percentile cutoff. Defaults to 98.0.

    Returns:
        np.ndarray: Contrast-stretched image scaled to [0, 255].
    """
    lo = np.percentile(gray_u8, np.clip(p_low, 0, 50))
    hi = np.percentile(gray_u8, np.clip(p_high, 50, 100))
    if hi <= lo + 1e-6:
        return gray_u8.copy()
    g = np.clip((gray_u8.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (g * 255.0 + 0.5).astype(np.uint8)

def canny_edges(gray_u8: np.ndarray, low: int = -1, high: int = -1, auto_sigma: float = 0.33) -> np.ndarray:
    """
    Detect edges using the Canny algorithm with optional thresholds.

    Args:
        gray_u8 (np.ndarray): Grayscale input image.
        low (int, optional): Lower Canny threshold. If negative, computed automatically. Defaults to -1.
        high (int, optional): Upper Canny threshold. If negative, computed automatically. Defaults to -1.
        auto_sigma (float, optional): Sigma for auto threshold calculation. Defaults to 0.33.

    Returns:
        np.ndarray: Binary edge map (uint8) with values in [0,255].
    """
    if low < 0 or high < 0:
        # get thresholds from median
        v = np.median(gray_u8)
        low = int(max(0, (1.0 - auto_sigma) * v))
        high = int(min(255, (1.0 + auto_sigma) * v))
    return cv2.Canny(gray_u8, low, high)

def rembg_dim_background(
    img_bgr: np.ndarray,
    *,
    dim_factor: float = 0.5,
    feather_px: int = 6,
    erode_px: int = 0
) -> np.ndarray:
    """
    Darken the image background using a segmentation mask from rembg.

    Uses the rembg library to estimate a foreground mask and darkens
    the background region by a specified dimming factor.

    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        dim_factor (float, optional): Amount to darken the background (0.0-1.0). Defaults to 0.5.
        feather_px (int, optional): Gaussian blur radius for mask feathering. Defaults to 6.
        erode_px (int, optional): Pixels to erode the foreground mask. Defaults to 0.

    Returns:
        np.ndarray: Input image with background dimmed. If rembg is unavailable, returns the original.
    """
    if not _HAS_REMBG:
        print("rembg not found")
        return img_bgr
    
    if (dim_factor <= 0.0):
        return img_bgr

    print("darkening backgound")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = rembg_remove(rgb)
    if out.ndim == 3 and out.shape[2] == 4:
        alpha = out[:, :, 3]
    else:
        return img_bgr

    # feathered background mask (1 = background, 0 = foreground)
    fg = (alpha.astype(np.float32) / 255.0)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        fg = cv2.erode((fg*255).astype(np.uint8), k, iterations=1).astype(np.float32)/255.0
    bg_mask = 1.0 - fg
    if feather_px > 0:
        bg_mask = cv2.GaussianBlur(bg_mask, (0,0), feather_px)

    # Darken background only
    bg_mask = bg_mask[..., None]
    dim = np.clip(float(dim_factor), 0.0, 1.0)
    scale = 1.0 - dim * bg_mask
    out_bgr = (img_bgr.astype(np.float32) * scale).clip(0,255).astype(np.uint8)
    return out_bgr

def apply_gamma(gray_u8: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to shift midtones.

    Args:
        gray_u8 (np.ndarray): Grayscale input image.
        gamma (float, optional): Gamma value (<1.0 brightens, >1.0 darkens). Defaults to 1.0.

    Returns:
        np.ndarray: Gamma-corrected image.
    """
    if abs(gamma - 1.0) < 1e-6:
        return gray_u8
    g = gray_u8.astype(np.float32) / 255.0
    g = np.power(g, gamma)
    return (g * 255.0 + 0.5).astype(np.uint8)

def brightness_clip(gray_u8: np.ndarray, clip_high: float = 98.0) -> np.ndarray:
    """
    Clip the brightest highlights based on percentile threshold.

    Args:
        gray_u8 (np.ndarray): Grayscale input image.
        clip_high (float, optional): Upper percentile cutoff. Defaults to 98.0.

    Returns:
        np.ndarray: Input image with clipped highlights.
    """
    hi = np.percentile(gray_u8, clip_high)
    if hi <= 1:
        return gray_u8
    g = np.clip(gray_u8.astype(np.float32) / hi, 0, 1)
    return (g * 255.0 + 0.5).astype(np.uint8)

def build_target_for_solver(
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
    use_rembg: bool,
    rembg_dim: float,
    rembg_feather: int,
    rembg_erode: int,
    pp_gamma: float,
    pp_clip_high: float,
) -> np.ndarray:
    """
    Build a preprocessed target image.

    Applies resizing, optional background dimming, CLAHE, contrast stretching,
    gamma correction, highlight clipping, and optional edge detection.

    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        work_size (int): Final image size for the solver.
        use_clahe (bool): Whether to apply CLAHE.
        use_contrast (bool): Whether to apply contrast stretching.
        p_low (float): Low percentile for contrast stretching.
        p_high (float): High percentile for contrast stretching.
        use_edges (bool): Whether to blend Canny edges into darkness map.
        edge_weight (float): Weight of edge influence.
        edge_low (int): Lower Canny threshold.
        edge_high (int): Upper Canny threshold.
        edge_auto_sigma (float): Sigma for automatic threshold estimation.
        use_rembg (bool): Whether to dim background using rembg.
        rembg_dim (float): Dimming factor for rembg background.
        rembg_feather (int): Mask feathering in pixels.
        rembg_erode (int): Mask erosion in pixels.
        pp_gamma (float): Gamma correction value.
        pp_clip_high (float): Brightness clipping percentile.

    Returns:
        np.ndarray: Preprocessed target brightness image
    """
    img = resize_square(img_bgr, work_size)

    if use_rembg and rembg_dim > 0.0:
        img = rembg_dim_background(
            img,
            dim_factor=rembg_dim,
            feather_px=int(rembg_feather),
            erode_px=int(rembg_erode),
        )

    gray = to_gray_u8(img)

    if use_clahe:
        gray = apply_clahe(gray)

    if use_contrast:
        gray = contrast_stretch(gray, p_low=p_low, p_high=p_high)

    if pp_gamma != 1.0:
        gray = apply_gamma(gray, pp_gamma)

    if pp_clip_high < 100.0:
        gray = brightness_clip(gray, clip_high=pp_clip_high)

    gray_f = gray.astype(np.float32) / 255.0
    darkness = 1.0 - gray_f

    if use_edges:
        e = canny_edges(gray, low=edge_low, high=edge_high, auto_sigma=edge_auto_sigma).astype(np.float32) / 255.0
        target_dark = (1.0 - edge_weight) * darkness + edge_weight * e
    else:
        target_dark = darkness

    src = (255.0 * (1.0 - target_dark)).astype(np.uint8)
    return src

def build_importance_map(gray: np.ndarray, worksize: int = 512) -> np.ndarray:
    """
    Compute an importance map for solver weighting of Faces/Foreground/Backgound.

    Combines rembg background detection and MediaPipe face detection to prioritize
    foreground and facial regions in the solver optimization.

    Args:
        gray (np.ndarray): Grayscale source image.
        worksize (int, optional): Output map size. Defaults to 512.

    Returns:
        np.ndarray: Float32 importance map (H, W) in range [0.1, 1.0].
    """
    # initialize weights:
    weight_fg = 0.8
    weight_bg = 0.2
    weight_face = 1.5

    H, W = gray.shape
    imp = np.ones((H, W), dtype=np.float32) * weight_fg

    try:
        from rembg import remove
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        out = remove(rgb)
        if out.ndim == 3 and out.shape[2] == 4:
            alpha = out[:, :, 3].astype(np.float32) / 255.0
            bg_mask = 1.0 - alpha
            imp[bg_mask > 0.5] = weight_bg
    except Exception:
        pass

    # face detection using MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        min_detection_confidence=0.5,
        refine_landmarks=False
    ) as face_mesh:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # collect all landmark points (normalized [0..1]) and convert to pixel coords
                pts = []
                for lm in face_landmarks.landmark:
                    # clamp coordinates to image
                    x = int(np.clip(lm.x, 0.0, 1.0) * (W - 1))
                    y = int(np.clip(lm.y, 0.0, 1.0) * (H - 1))
                    pts.append((x, y))

                pts_arr = np.array(pts, dtype=np.int32)
                if pts_arr.shape[0] >= 3:
                    hull = cv2.convexHull(pts_arr)
                    cv2.fillConvexPoly(imp, hull, weight_face)

    # resize to solver worksize and smooth slightly
    imp_resized = cv2.resize(imp, (worksize, worksize), interpolation=cv2.INTER_LINEAR)
    # slight gaussian blur to avoid very hard edges
    imp_resized = cv2.GaussianBlur(imp_resized, (0, 0), sigmaX=2.0)

    # clamp to sensible bounds [0.1, 1.0], ensure background at least 0.1
    imp_resized = np.clip(imp_resized, 0.1, 1.0).astype(np.float32)

    return imp_resized