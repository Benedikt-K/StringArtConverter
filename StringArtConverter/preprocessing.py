from __future__ import annotations
from rembg import remove as rembg_remove
import cv2
import numpy as np

def circular_mask(h: int, w: int, margin: int = 16) -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = min(cx, cy) - margin
    return (((x - cx)**2 + (y - cy)**2) <= (r*r)).astype(np.float32)

def to_grayscale(img_bgr: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """
    Convert to grayscale (uint8 0..255) with blur. Optionally apply CLAHE for better contrast.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    else:
        gray = cv2.equalizeHist(gray)
    # apply blur to add micro-contrasts
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0) # 1.5*gray - 0.5*blur
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp

def detect_edges_canny(gray: np.ndarray, low: int = 80, high: int = 200, blur: int = 3) -> np.ndarray:
    """
    Edge map in float32 [0,1].
    - low/high: Canny thresholds
    - blur: kernel size for Gaussian blur to thicken edges (0 disables)
    """
    edges = cv2.Canny(gray, low, high).astype(np.float32) / 255.0
    if blur > 0:
        edges = cv2.GaussianBlur(edges, (blur, blur), 0)
    return edges

def detect_edges_sobel(gray: np.ndarray, blur: int = 3) -> np.ndarray:
    """
    Softer edge map in [0,1] using gradient magnitude (Sobel).
    Better for soft structure like skin/eyelids/lashes.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if blur > 0:
        mag = cv2.GaussianBlur(mag, (blur, blur), 0)
    mag /= (mag.max() + 1e-6)
    return mag

def build_target(
    gray: np.ndarray,
    edge_weight: float = 0.60,
    tone_weight: float = 0.20,
    edge_mode: str = "sobel",
    *,
    edge_blur_ksize: int = 3,
    edge_high_pct: float = 0.90,   # 90th percentile for stretching edges
    tone_gamma: float = 1.0,       # >1 darkens mids, <1 brightens mids
) -> np.ndarray:
    """
    Produce target in [0,1] as an explicit 3-way mix:
      target = w_dark * dark + w_edge * edges + w_tone * tone
    with w_dark = 1 - (edge_weight + tone_weight).
    Extra steps: edge denoise+stretch, keep bright zones slightly >0, optional tone gamma.
    """
    # --- base maps ---
    tone = gray.astype(np.float32) / 255.0

    # darkness; keep bright zones slightly >0 so they aren't completely ignored
    dark = (1.0 - tone)
    dark = np.clip(dark, 0.0, 0.92)

    # edges
    if edge_mode.lower() == "canny":
        edges = detect_edges_canny(gray, low=70, high=180, blur=edge_blur_ksize)
    else:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(gx, gy)
        if edge_blur_ksize > 0:
            edges = cv2.GaussianBlur(edges, (edge_blur_ksize, edge_blur_ksize), 0)
        # contrast-stretch by percentile to suppress noise
        hi = float(np.percentile(edges, edge_high_pct * 100.0))
        lo = float(np.percentile(edges, 40.0))  # floor so weak texture goes to ~0
        edges = (edges - lo) / (max(1e-6, hi - lo))
        edges = np.clip(edges, 0.0, 1.0)

    # optional tone gamma for portraits
    if abs(tone_gamma - 1.0) > 1e-6:
        tone = np.power(tone, tone_gamma)

    # --- explicit 3-way weights (sum to 1) ---
    ew = float(np.clip(edge_weight, 0.0, 1.0))
    tw = float(np.clip(tone_weight, 0.0, 1.0))
    dw = max(0.0, 1.0 - (ew + tw))  # remaining mass to darkness

    target = dw * dark + ew * edges + tw * tone
    return np.clip(target, 0.0, 1.0)

def resize(
    img_bgr: np.ndarray,
    size: int = 512,
    mode: str = "cover",        # "cover" center-crops; "contain" pads (your old behavior)
    pad_value: int = 255,
) -> np.ndarray:
    """
    Resize to a square (size x size).

    - mode="cover": scale so the *smaller* side matches `size`, then center-crop the larger side.
      (fills the square with image pixels; no borders)
    - mode="contain": scale so the *larger* side matches `size`, then pad with `pad_value`.
      (keeps whole image visible; may leave borders)
    """
    h0, w0 = img_bgr.shape[:2]
    if mode.lower() == "contain":
        scale = size / max(h0, w0)
        new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
        y_off = (size - new_h) // 2
        x_off = (size - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    # cover: scale so the smaller side fits, then center-crop the bigger side
    scale = size / min(h0, w0)
    new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # center-crop to size x size
    y_start = max(0, (new_h - size) // 2)
    x_start = max(0, (new_w - size) // 2)
    cropped = resized[y_start:y_start + size, x_start:x_start + size]
    # in rare rounding cases cropped may be off by 1px; enforce exact size
    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return cropped

def remove_background(img_bgr: np.ndarray) -> np.ndarray:
    result = rembg_remove(img_bgr)
    # drop alpha and replace with white where transparent
    if result.shape[2] == 4:
        alpha = result[:, :, 3] / 255.0
        fg = result[:, :, :3]
        white_bg = np.ones_like(fg, dtype=np.uint8) * 255
        blended = (fg * alpha[..., None] + white_bg * (1 - alpha[..., None])).astype(np.uint8)
        return blended
    return result

def preview(
    img_bgr: np.ndarray,
    edge_weight: float = 0.5,
    show: bool = True,
    save_path: str | None = None,
    max_size: int = 512,
) -> None:
    """
    Generate grayscale, edges, and target preview.
    - show: open in a resizable OpenCV window
    - save_path: if given, save preview as PNG
    - max_size: maximum size of the image
    """
    img_bgr = resize(img_bgr, size=max_size)
    #img_bgr = remove_background(img_bgr)
    gray = to_grayscale(img_bgr, use_clahe=True)
    edges = detect_edges_canny(gray)
    target = build_target(gray, edge_weight=edge_weight)

    # convert to displayable 0..255 uint8
    gray_disp = gray
    edges_disp = (edges * 255).astype(np.uint8)
    target_disp = (target * 255).astype(np.uint8)

    # make 3-channel so we can stack horizontally
    def to3(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack([to3(gray_disp), to3(edges_disp), to3(target_disp)])

    return stacked