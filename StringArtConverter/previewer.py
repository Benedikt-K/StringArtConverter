import numpy as np
import cv2

def render_path(work_size, 
                pins, 
                path, 
                alpha_per_line=0.10,
                gamma=1.0, 
                thickness=1, 
                margin=16):
    """
    Render preview image from path with tunable darkness and gamma.
    """
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
