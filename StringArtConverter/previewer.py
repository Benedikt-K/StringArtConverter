import numpy as np
import cv2
from typing import Tuple, Iterable
from .pin_solver import pin_positions_circle

Color = Tuple[int, int, int]  # RGB

def _line_to_overlay_and_mask(
    shape: Tuple[int, int],
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    color: Color,
    thickness: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      overlay_rgb: HxWx3 float32 in [0,1] with the line color on black
      mask:        HxWx1 float32 in {0,1} (1 where the line exists)
    """
    h, w = shape
    # draw BGR for OpenCV
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    line_img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.line(
        line_img,
        (int(p0[0]), int(p0[1])),
        (int(p1[0]), int(p1[1])),
        color=bgr,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_AA,
    )
    overlay_rgb = line_img[:, :, ::-1].astype(np.float32) / 255.0  # to RGB [0,1]
    # any channel >0 means part of the line; make a 1-channel mask in {0,1}
    mask = (overlay_rgb.max(axis=2, keepdims=True) > 0).astype(np.float32)
    return overlay_rgb, mask

def _masked_blend(canvas: np.ndarray, overlay_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> None:
    """
    In-place masked alpha blend only where mask==1:
      canvas = canvas*(1 - a) + overlay*a  on masked pixels
    """
    a = np.clip(alpha, 0.0, 1.0)
    # per-pixel alpha: a on line, 0 elsewhere
    A = a * mask
    np.multiply(canvas, (1.0 - A), out=canvas)
    canvas += overlay_rgb * A

def simulate_string_art(
    width: int,
    height: int,
    n_pins: int,
    segments: Iterable[Tuple[int, int]],
    *,
    margin: int = 16,
    background_color: Color = (245, 245, 245),
    thread_color: Color = (30, 30, 30),
    thickness: int = 1,
    alpha_per_segment: float = 0.08,
) -> np.ndarray:
    bg = np.array(background_color, dtype=np.float32) / 255.0
    canvas = np.empty((height, width, 3), dtype=np.float32); canvas[:] = bg

    pins = pin_positions_circle(width, height, n_pins, margin=margin)

    for (i, j) in segments:
        if not (0 <= i < n_pins and 0 <= j < n_pins) or i == j:
            continue
        p0 = (int(pins[i, 0]), int(pins[i, 1]))
        p1 = (int(pins[j, 0]), int(pins[j, 1]))
        overlay_rgb, mask = _line_to_overlay_and_mask((height, width), p0, p1, thread_color, thickness)
        _masked_blend(canvas, overlay_rgb, mask, alpha_per_segment)

    # optional: draw nails
    for (x, y) in pins: cv2.circle(canvas, (int(x), int(y)), 2, (0,0,0), -1, lineType=cv2.LINE_AA)

    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)