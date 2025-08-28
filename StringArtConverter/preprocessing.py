from __future__ import annotations
import cv2
import numpy as np

try:
    from rembg import remove as rembg_remove
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False

def resize_square(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """
    Resize image to the desired size
    """
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

def to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale
    """
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

def rembg_dim_background(
    img_bgr: np.ndarray,
    *,
    dim_factor: float = 0.5,
    feather_px: int = 6,
    erode_px: int = 0
) -> np.ndarray:
    """
    Use rembg to get a foreground mask, darken ONLY the background by dim_factor.
    Feather_px softens the mask edges, erode_px shriks the mask
    Returns a BGR image with a darker bg, same size as input.

    If rembg is unavailable, returns the original image unchanged.
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
    Shifts midtones of the image

    gamma < 1.0 brightens midtones, gamma > 1.0 darkens midtones
    """
    if abs(gamma - 1.0) < 1e-6:
        return gray_u8
    g = gray_u8.astype(np.float32) / 255.0
    g = np.power(g, gamma)
    return (g * 255.0 + 0.5).astype(np.uint8)

def brightness_clip(gray_u8: np.ndarray, clip_high: float = 98.0) -> np.ndarray:
    """
    Clip brightest highlights off
    """
    hi = np.percentile(gray_u8, clip_high)
    if hi <= 1:
        return gray_u8
    g = np.clip(gray_u8.astype(np.float32) / hi, 0, 1)
    return (g * 255.0 + 0.5).astype(np.uint8)

def build_brightness_for_solver(
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
    Builds the target using specified preprocessing steps and settings

    Internally we form a 'target darkness' in [0..1], then convert:
      SourceImage_u8 = 255 * (1 - target_darkness)
    so dark/edgey areas become low brightness, high error (255 - src).

    Returns uint8 brightness image (H,W) where 0=black, 255=white.
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
