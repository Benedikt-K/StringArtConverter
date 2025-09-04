import numpy as np
import cv2

def render_path(work_size,
                pins,
                path,
                alpha_per_line=0.10,
                gamma=1.0,
                thickness=1,
                margin=16):

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

    for a, b in path:
        x0, y0 = int(pins_xy[a, 0]), int(pins_xy[a, 1])
        x1, y1 = int(pins_xy[b, 0]), int(pins_xy[b, 1])

        tmp.fill(0)

        # draw the line into tmp
        cv2.line(tmp, (x0, y0), (x1, y1), alpha_per_line,
                 thickness=thickness, lineType=cv2.LINE_AA)

        cv2.accumulate(tmp, acc, mask=board_mask)

    np.clip(acc, 0.0, 1.0, out=acc)
    if gamma != 1.0:
        inv = 1.0 / gamma
        cv2.pow(acc, inv, acc)

    img = (255.0 * (1.0 - acc)).astype(np.uint8)
    return img