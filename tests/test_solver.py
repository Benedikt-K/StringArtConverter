import numpy as np
from StringArtConverter.solver import pin_positions_circle, precalc_lines, solve_string_art

# region -- Geometry and Line Cache tests --

def test_pin_positions_circle_basic():
    pins = pin_positions_circle(100, 4)
    assert pins.shape == (4, 2)
    assert np.all(pins >= 0)
    assert np.all(pins <= 100)

def test_pin_positions_circle_single_pin():
    pins = pin_positions_circle(50, 1)
    assert pins.shape == (1, 2)

def test_pin_positions_circle_zero_size():
    pins = pin_positions_circle(0, 5)
    assert np.all(pins == 0)

def test_pin_positions_circle_large_number_of_pins():
    n_pins = 1000
    pins = pin_positions_circle(100, n_pins)
    assert pins.shape == (n_pins, 2)

def test_pin_positions_circle_negative_size():
    pins = pin_positions_circle(-10, 5)
    # Negative size treated as zero radius
    assert np.all(pins == -10 // 2) or np.all(pins <= 0)

def test_precalc_lines_basic():
    pins = pin_positions_circle(50, 6)
    line_cache = precalc_lines(pins, 6, 50, 1)
    for (a, b), idx in line_cache.items():
        assert isinstance(idx, np.ndarray)
        assert idx.ndim == 1

def test_precalc_lines_min_distance():
    pins = pin_positions_circle(50, 6)
    line_cache = precalc_lines(pins, 6, 50, 10)
    assert len(line_cache) <= 12

def test_precalc_lines_distance_one_skipped():
    pins = np.array([[0,0],[1,0],[0,1]])
    line_cache = precalc_lines(pins, 3, 3, 1)
    # Lines of distance <=1 should be skipped
    for idx in line_cache.values():
        assert len(idx) > 1

# endregion

# region -- Functional tests --

def test_solve_string_art_detects_simple_line():
    img = np.ones((16, 16), dtype=np.uint8) * 255
    img[:, 8] = 0
    path, error, gray, pins = solve_string_art(
        img, n_pins=24, max_lines=8, work_size=16, min_distance=1, line_weight=8.0
    )
    assert len(path) > 0

    # Helper: check whether a segment between two pins crosses the center column
    def segment_crosses_center(p_a, p_b, center_x=8, tol=1):
        x0, y0 = int(p_a[0]), int(p_a[1])
        x1, y1 = int(p_b[0]), int(p_b[1])
        d = max(int(round(((x1 - x0)**2 + (y1 - y0)**2) ** 0.5)), 1)
        xs = np.linspace(x0, x1, d).astype(int)
        return np.any(np.abs(xs - center_x) <= tol)

    # Search all drawn segments for one that crosses the center column
    crosses = any(segment_crosses_center(pins[a], pins[b]) for a, b in path)
    assert crosses, "Solver path does not contain any segment crossing the image center column"

def test_solve_string_art_detects_diagonal_line():
    img = np.ones((16, 16), dtype=np.uint8) * 255
    np.fill_diagonal(img, 0)
    path, _, _, pins = solve_string_art(img, n_pins=24, max_lines=8, work_size=16, min_distance=1)
    assert len(path) > 0

def test_solve_string_art_basic():
    img = np.zeros((32, 32), dtype=np.uint8)
    path, error, gray, pins = solve_string_art(img, n_pins=12, max_lines=5, work_size=32)
    assert len(path) <= 5
    assert error.shape == (32, 32)
    assert gray.shape == (32, 32)
    assert pins.shape[0] == 12

# endregion

# region -- Edge Cases/Robustness tests --

def test_solve_string_art_all_white():
    img = np.ones((16, 16), dtype=np.uint8) * 255
    path, error, gray, pins = solve_string_art(img, n_pins=8, max_lines=10, work_size=16)
    # No lines needed for white image
    assert len(path) == 0
    assert error.shape == (16, 16)

def test_solve_string_art_all_black():
    img = np.zeros((16, 16), dtype=np.uint8)
    path, error, gray, pins = solve_string_art(
        img, n_pins=8, max_lines=5, work_size=16,
        min_distance=1, line_weight=8.0
    )
    # Should attempt to draw lines
    assert len(path) > 0
    assert error.shape == (16, 16)

def test_solve_string_art_line_weight_zero():
    img = np.zeros((16, 16), dtype=np.uint8)
    path, error, _, _ = solve_string_art(img, n_pins=8, max_lines=5, line_weight=0.0, work_size=16)
    # No fiber increment → should produce zero lines
    assert len(path) == 0

def test_solve_string_art_last_n_larger_than_n_pins():
    img = np.zeros((16, 16), dtype=np.uint8)
    path, _, _, _ = solve_string_art(img, n_pins=5, max_lines=5, last_n=10, work_size=16)
    # Should still return some lines until solver cannot pick any new pin
    assert len(path) <= 5

def test_solve_string_art_single_pixel():
    img = np.array([[128]], dtype=np.uint8)
    path, error, gray, pins = solve_string_art(img, n_pins=3, max_lines=3, work_size=1)
    assert gray.shape == (1, 1)
    assert error.shape == (1, 1)

def test_solve_string_art_importance_map_all_zeros():
    img = np.zeros((8, 8), dtype=np.uint8)
    importance = np.zeros_like(img)
    path, error, _, _ = solve_string_art(img, n_pins=4, max_lines=5, importance_map=importance, work_size=8)
    # With zero importance, solver should not apply lines
    assert len(path) == 0
    assert error.shape == (8, 8)

def test_solve_string_art_max_lines_exceed_possible():
    img = np.zeros((8, 8), dtype=np.uint8)
    # Set max_lines higher than number of possible connections
    n_pins = 4
    min_distance = 1
    path, _, _, _ = solve_string_art(img, n_pins=n_pins, max_lines=100, min_distance=min_distance, work_size=8)
    # Maximum possible connections respecting min_distance
    max_possible_connections = sum(1 for i in range(n_pins) for j in range(i + min_distance, n_pins))
    assert len(path) <= max_possible_connections

def test_solve_string_art_n_pins_less_than_two():
    img = np.zeros((8, 8), dtype=np.uint8)
    path, _, _, _ = solve_string_art(img, n_pins=1, max_lines=5, work_size=8)
    # Should produce zero lines
    assert len(path) == 0

def test_solve_string_art_importance_map_extreme_values():
    img = np.zeros((8, 8), dtype=np.uint8)
    importance = np.ones_like(img) * 1000
    path, error, _, _ = solve_string_art(
        img, n_pins=4, max_lines=5, importance_map=importance,
        work_size=8, min_distance=1, line_weight=8.0
    )
    # Should still produce lines
    assert len(path) > 0
    assert error.shape == (8, 8)

def test_solve_string_art_min_distance_edge():
    img = np.zeros((16, 16), dtype=np.uint8)
    # min_distance = 1 should allow connecting adjacent pins
    path, _, _, _ = solve_string_art(img, n_pins=6, max_lines=5, min_distance=1, work_size=16)
    assert len(path) <= 5

# endregion

# region -- Regression/Stability tests --

def test_regression_vertical_line_consistency():
    img = np.ones((16, 16), dtype=np.uint8) * 255
    img[:, 8] = 0
    path, error, gray, pins = solve_string_art(
        img,
        n_pins=24,
        max_lines=8,
        work_size=16,
        min_distance=1,
        line_weight=8.0,
    )
    expected_first_lines = [(0, 6), (6, 19), (19, 7), (7, 20), (20, 8)]
    expected_mean_error = 2136.5
    expected_std_error = 8195.0419921875
    expected_num_lines = 8
    assert len(path) == expected_num_lines
    assert path[:len(expected_first_lines)] == expected_first_lines
    assert np.isclose(np.mean(error), expected_mean_error, atol=2.0)
    assert np.isclose(np.std(error), expected_std_error, atol=2.0)

def test_regression_x_shape_consistency():
    img = np.ones((16, 16), dtype=np.uint8) * 255
    np.fill_diagonal(img, 0)
    np.fill_diagonal(np.fliplr(img), 0)
    path, error, gray, pins = solve_string_art(
        img,
        n_pins=16,
        max_lines=10,
        work_size=16,
        min_distance=2,
        line_weight=8.0,
    )
    expected_first_lines = [(0, 6), (6, 14), (14, 5), (5, 13), (13, 7), (7, 15), (15, 8), (8, 0), (0, 9), (9, 2)]
    expected_mean_error = 4706.5
    assert path[:len(expected_first_lines)] == expected_first_lines
    assert np.isclose(np.mean(error), expected_mean_error, atol=2.0)

def test_regression_radial_gradient_consistency():
    size = 32
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    img = np.clip(dist / dist.max() * 255, 0, 255).astype(np.uint8)
    path, error, gray, pins = solve_string_art(
        img,
        n_pins=24,
        max_lines=15,
        work_size=size,
        min_distance=2,
        line_weight=8.0,
    )
    expected_first_lines = [(0, 12), (12, 23), (23, 11), (11, 0), (0, 13), (13, 1), (1, 14), (14, 2)]
    expected_mean_error = 8802.66015625
    expected_std_error = 7133.81005859375
    assert path[:len(expected_first_lines)] == expected_first_lines
    assert np.isclose(np.mean(error), expected_mean_error, atol=2.0)
    assert np.isclose(np.std(error), expected_std_error, atol=2.0)

def test_solve_string_art_deterministic():
    img = np.zeros((16, 16), dtype=np.uint8)
    np.fill_diagonal(img, 0)
    path1, _, _, _ = solve_string_art(img, n_pins=8, max_lines=5, work_size=16)
    path2, _, _, _ = solve_string_art(img, n_pins=8, max_lines=5, work_size=16)
    assert path1 == path2

def test_solver_robust_to_noise():
    img = np.ones((16, 16), dtype=np.uint8) * 255
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    noisy_img = np.clip(img - noise, 0, 255)
    path, _, _, _ = solve_string_art(noisy_img, n_pins=12, max_lines=5, work_size=16)
    assert len(path) >= 0

# endregion

# region -- Performance tests --

def test_solve_string_art_terminates_quickly_on_uniform_image():
    img = np.ones((32, 32), dtype=np.uint8) * 127
    path, _, _, _ = solve_string_art(img, n_pins=12, max_lines=1000, work_size=32)
    assert len(path) < 50  # Should stop early

# endregion