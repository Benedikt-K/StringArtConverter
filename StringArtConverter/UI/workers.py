import cv2
import numpy as np
import os
from itertools import product

from PySide6.QtCore import Signal, QObject

from StringArtConverter.preprocessing import build_target_for_solver, to_gray_u8, build_importance_map
from StringArtConverter.previewer import render_path
from StringArtConverter.solver import solve_string_art
from StringArtConverter.previewer import visualize_importance_map, visualize_target


# region ----------- convert worker -----------
class ConvertWorker(QObject):
    """
    Worker class that performs the string art conversion process in a background thread.

    Signals:
        progress (int): Emitted with progress percentage (0-100).
        finished (list, np.ndarray, np.ndarray, np.ndarray): Emitted when the conversion
            is complete, returning: the calculated path, error map, target, and pin positions.
        errored (str): Emitted with an error message if an exception occurs.
    """
    progress = Signal(int)
    finished = Signal(list, np.ndarray, np.ndarray, np.ndarray)  # path, error, target, pins
    errored = Signal(str)

    def __init__(self, img_bgr: np.ndarray, params: dict):
        """
        Args:
            img_bgr (np.ndarray): Input image in BGR format.
            params (dict): Dictionary of solver and preprocessing parameters.
        """
        super().__init__()
        self.img_bgr = img_bgr
        self.params = params

    def run(self):
        """
        Run the string art conversion process.

        Emits:
            - 'progress(int)': periodically with progress percentage.
            - 'finished(list, np.ndarray, np.ndarray, np.ndarray)': upon success.
            - 'errored(str)': upon exception.
        """
        try:
            src_u8 = build_target_for_solver(
                img_bgr=self.img_bgr,
                work_size=self.params["work_size"],
                use_clahe=self.params["pp_clahe"],
                use_contrast=self.params["pp_contrast"],
                p_low=self.params["pp_c_low"],
                p_high=self.params["pp_c_high"],
                use_edges=self.params["pp_edges"],
                edge_weight=self.params["pp_edge_weight"],
                edge_low=self.params["pp_edge_low"],
                edge_high=self.params["pp_edge_high"],
                edge_auto_sigma=self.params["pp_edge_auto_sigma"],
                use_rembg=self.params["pp_rembg"],
                rembg_dim=self.params["pp_rembg_dim"],
                rembg_feather=self.params["pp_rembg_feather"],
                rembg_erode=self.params["pp_rembg_erode"],
                pp_gamma=self.params.get("pp_gamma", 1.0),
                pp_clip_high=self.params.get("pp_clip_high", 100.0),
            )

            def cb(p: int):
                self.progress.emit(p)

            # get importance map, gray unnötig?
            gray = to_gray_u8(self.img_bgr)
            importance = build_importance_map(gray, worksize=self.params["work_size"])

            # debug outputs
            """
            imp_vis = visualize_importance_map(importance)
            cv2.imwrite("importance_debug.png", imp_vis)

            target_vis = visualize_target(gray)
            cv2.imwrite("target_debug.png", target_vis)
            """

            path, err, target, pins = solve_string_art(
                source_brightness_u8=src_u8,
                n_pins=self.params["pins"],
                max_lines=self.params["steps"],
                min_distance=self.params["min_distance"],
                line_weight=self.params["line_weight"],
                last_n=self.params["last_n"],
                work_size=self.params["work_size"],
                importance_map=importance,
                progress_cb=cb,
            )
            self.finished.emit(path, err, target, pins)
        except Exception as e:
            self.errored.emit(str(e))
# endregion

# region ----------- param search worker -----------
class BatchSearchWorker(QObject):
    """
    Worker that performs grid-based parameter search for batch evaluation.

    Note: this is computationaly intensive, especially when using a lot of possible
    parameter combinations, as the problem is solved for all of them.

    Signals:
        progress (int): Emitted with progress percentage (0-100).
        finished (str): Emitted when batch search completes successfully.
        errored (str): Emitted with an error message if an exception occurs.
    """
    progress = Signal(int)
    finished = Signal(str)
    errored = Signal(str)

    def __init__(self, img_bgr: np.ndarray, base_params: dict, grid: dict, out_dir: str):
        """
        Args:
            img_bgr (np.ndarray): Input image in BGR format.
            base_params (dict): Base configuration parameters.
            grid (dict): Dictionary mapping parameter names to lists of possible values.
            out_dir (str): Directory where result images and logs are saved.
        """
        super().__init__()
        self.img_bgr = img_bgr
        self.base = base_params
        self.grid = grid
        self.out_dir = out_dir

    def _variants(self):
        """
        Generate all parameter combinations from the grid.

        Yields:
            dict: A parameter dictionary combining base parameters with one combination
                of grid values.
        """
        keys = list(self.grid.keys())
        vals = [self.grid[k] for k in keys]
        for combo in product(*vals):
            p = dict(self.base)
            for k, v in zip(keys, combo):
                p[k] = v
            yield p

    def run(self):
        """
        Execute the batch grid search and save all results.

        For each parameter combination, this worker:
        - Preprocesses the input image.
        - Solves the string art pattern.
        - Renders a preview image.
        - Saves results and metrics to disk.

        Emits:
            - 'progress(int)': periodically with completion percentage.
            - 'finished(str)': with output directory path when done.
            - 'errored(str)': with error message on exception.
        """
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            lines = []
            total = 0
            for _ in self._variants():
                total += 1
            if total == 0:
                self.errored.emit("Grid is empty.")
                return

            idx = 0
            for params in self._variants():
                idx += 1

                src_u8 = build_target_for_solver(
                    img_bgr=self.img_bgr,
                    work_size=params["work_size"],
                    use_clahe=params["pp_clahe"],
                    use_contrast=params["pp_contrast"],
                    p_low=params["pp_c_low"],
                    p_high=params["pp_c_high"],
                    use_edges=params["pp_edges"],
                    edge_weight=params["pp_edge_weight"],
                    edge_low=-1, edge_high=-1, edge_auto_sigma=0.33,
                    use_rembg=params["pp_rembg"],
                    rembg_dim=params["pp_rembg_dim"],
                    rembg_feather=params["pp_rembg_feather"],
                    rembg_erode=params["pp_rembg_erode"],
                    pp_gamma=params.get("pp_gamma", 1.0),
                    pp_clip_high=params.get("pp_clip_high", 100.0),
                )

                path, err, target, pins = solve_string_art(
                    source_brightness_u8=src_u8,
                    n_pins=params["pins"],
                    max_lines=params["steps"],
                    min_distance=params["min_distance"],
                    line_weight=params["line_weight"],
                    last_n=params["last_n"],
                    work_size=params["work_size"],
                    progress_cb=None,
                )

                preview = render_path(
                    work_size=params["work_size"],
                    pins=pins,
                    path=path,
                    alpha_per_line=params["render_alpha"],
                    gamma=params["render_gamma"],
                    thickness=params["line_thickness"],
                )

                img_name = f"{idx}.png"
                cv2.imwrite(os.path.join(self.out_dir, img_name), preview)

                score = float(err.mean()) if err is not None else float("nan")

                # record only relevant params
                rec_keys = [
                    "work_size","pins","steps","min_distance","line_weight","last_n",
                    "pp_clahe","pp_contrast","pp_c_low","pp_c_high",
                    "pp_edges","pp_edge_weight",
                    "pp_rembg","pp_rembg_dim","pp_rembg_feather","pp_rembg_erode",
                    "pp_gamma","pp_clip_high",
                    "render_alpha","render_gamma","line_thickness"
                ]
                snapshot = ", ".join(f"{k}={params.get(k)}" for k in rec_keys)
                lines.append(f"{idx}: {img_name} | lines={len(path)} | score={score:.6f} | {snapshot}")

                self.progress.emit(int(100 * idx / total))

            with open(os.path.join(self.out_dir, "params.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            self.finished.emit(self.out_dir)

        except Exception as e:
            self.errored.emit(str(e))
# endregion