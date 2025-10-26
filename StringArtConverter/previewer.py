import numpy as np
import cv2

def render_path(work_size,
                pins,
                path,
                alpha_per_line=0.10,
                gamma=1.0,
                thickness=1,
                margin=16) -> np.ndarray:
    """
    Render a grayscale preview of the string art path.

    Accumulates all lines from the given pin sequence to simulate the final
    string pattern. Each line darkens the background by a small amount
    (alpha_per_line). The result approximates the visual appearance of
    the final thread composition.

    Args:
        work_size (int): Image size (assumed square) for the rendered preview.
        pins (np.ndarray): Array of pin coordinates with shape (n_pins, 2).
        path (list): Sequence of (start_pin, end_pin) pairs forming the path.
        alpha_per_line (float, optional): Darkness contribution per line segment.
            Defaults to 0.10.
        gamma (float, optional): Gamma correction applied after accumulation.
            Values < 1.0 brighten midtones; > 1.0 darken them. Defaults to 1.0.
        thickness (int, optional): Line thickness in pixels. Defaults to 1.
        margin (int, optional): Margin between the outer circle and the border.
            Used to define the valid circular area. Defaults to 16.

    Returns:
        np.ndarray: Rendered grayscale image, where darker regions correspond
        to more thread intersections.
    """
    H = W = int(work_size)
    acc = np.zeros((H, W), dtype=np.float32)

    # Precompute circular mask once
    yy, xx = np.ogrid[:H, :W]
    cx, cy = W * 0.5, H * 0.5
    r = min(H, W) * 0.5 - margin
    board_mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8)

    tmp = np.zeros_like(acc, dtype=np.float32)

    pins_xy = np.asarray(pins, dtype=np.float32)
    pins_xy = np.round(pins_xy).astype(np.int32)

    count = 0
    for a, b in path:
        x0, y0 = int(pins_xy[a, 0]), int(pins_xy[a, 1])
        x1, y1 = int(pins_xy[b, 0]), int(pins_xy[b, 1])

        tmp.fill(0)

        adaptive_alpha = alpha_per_line
        # commented out is different thread "colors" option
        '''
        # test other colors
        if count > 3000:
            adaptive_alpha *= 2
        if count > 3500:
            adaptive_alpha *= 2
            '''

        # draw the line into tmp
        cv2.line(tmp, (x0, y0), (x1, y1), adaptive_alpha,
                 thickness=thickness, lineType=cv2.LINE_AA)

        cv2.accumulate(tmp, acc, mask=board_mask)
        # test other colors
        count += 1

    np.clip(acc, 0.0, 1.0, out=acc)
    if gamma != 1.0:
        inv = 1.0 / gamma
        cv2.pow(acc, inv, acc)

    img = (255.0 * (1.0 - acc)).astype(np.uint8)
    return img

def visualize_importance_map(importance: np.ndarray) -> np.ndarray:
    """
    Visualize an importance map as an RGB color image.

    Maps importance values to distinct colors for visual debugging:
    - Low (background) = Blue
    - Medium (foreground) = Green
    - High (face) = Red

    Args:
        importance (np.ndarray): Importance map with values in [0, 1].

    Returns:
        np.ndarray: RGB visualization.
    """
    vis = np.zeros((*importance.shape, 3), dtype=np.uint8)
    norm = np.clip(importance, 0, 1)

    # Background ~0.2
    vis[norm < 0.4] = (255, 0, 0)  # Blue

    # Mid ~0.8
    midmask = (norm >= 0.4) & (norm < 1.0)
    vis[midmask] = (0, 255, 0)  # Green

    # High ~1.0
    vis[norm >= 1.0] = (0, 0, 255)  # Red

    return vis

def visualize_target(gray: np.ndarray, max_fibers_per_pixel: float = 200.0) -> np.ndarray:
    """
    Visualize the target density map as a grayscale image.

    Converts the inverted brightness map to a visualization where darker pixels
    indicate regions requiring more threads and lighter pixels indicate fewer.

    Args:
        gray (np.ndarray): Grayscale target brightness image.
        max_fibers_per_pixel (float, optional): Scaling factor used to represent 
        maximum fiber density. Defaults to 200.0.

    Returns:
        np.ndarray: Normalized grayscale visualization.
    """
    target_fibers = (1.0 - gray.astype(np.float32) / 255.0) * max_fibers_per_pixel
    target_norm = (target_fibers / max_fibers_per_pixel * 255).astype(np.uint8)
    return target_norm