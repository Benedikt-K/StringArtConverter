import numpy as np
import pytest
import warnings

import StringArtConverter.preprocessing as pp

# region -- Fixtures -- 

@pytest.fixture
def dummy_bgr():
    """Simple 8x10 color image."""
    h, w = 8, 10
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :w//2] = [255, 0, 0]   # blue left
    img[:, w//2:] = [0, 255, 0]   # green right
    return img


@pytest.fixture
def dummy_gray():
    """Simple 8x8 gradient grayscale image."""
    return np.tile(np.arange(8, dtype=np.uint8), (8, 1))

# endregion

# region -- Core tests --

def test_resize_square_returns_correct_shape(dummy_bgr):
    out = pp.resize_square(dummy_bgr, 16)
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8


def test_to_gray_u8_converts_channels(dummy_bgr):
    gray = pp.to_gray_u8(dummy_bgr)
    assert len(gray.shape) == 2
    assert gray.dtype == np.uint8


def test_apply_clahe_changes_contrast(dummy_gray):
    enhanced = pp.apply_clahe(dummy_gray)
    assert enhanced.shape == dummy_gray.shape
    assert not np.array_equal(enhanced, dummy_gray)


def test_contrast_stretch_increases_range(dummy_gray):
    stretched = pp.contrast_stretch(dummy_gray)
    assert stretched.min() == 0
    assert stretched.max() == 255


def test_canny_edges_auto_threshold(dummy_gray):
    edges = pp.canny_edges(dummy_gray)
    assert edges.shape == dummy_gray.shape
    assert edges.dtype == np.uint8


@pytest.mark.parametrize("gamma,expect_change", [(1.0, False), (0.5, True), (2.0, True)])
def test_apply_gamma(dummy_gray, gamma, expect_change):
    out = pp.apply_gamma(dummy_gray, gamma)
    if expect_change:
        assert not np.array_equal(out, dummy_gray)
    else:
        assert np.array_equal(out, dummy_gray)


def test_brightness_clip_limits_highlights(dummy_gray):
    clipped = pp.brightness_clip(dummy_gray, clip_high=90.0)
    assert clipped.max() <= 255
    assert clipped.min() >= 0

# endregion

# region -- Edge cases --

def test_contrast_stretch_handles_flat_image():
    flat = np.ones((8, 8), dtype=np.uint8) * 100
    out = pp.contrast_stretch(flat)
    assert np.array_equal(flat, out)


def test_handle_missing_rembg_dim_background(monkeypatch, dummy_bgr):
    monkeypatch.setattr(pp, "_HAS_REMBG", False)
    out = pp.rembg_dim_background(dummy_bgr)
    assert np.array_equal(out, dummy_bgr)

# endregion

# region -- Integration tests --

def test_build_target_for_solver_basic(dummy_bgr):
    result = pp.build_target_for_solver(
        dummy_bgr,
        work_size=16,
        use_clahe=True,
        use_contrast=True,
        p_low=2,
        p_high=98,
        use_edges=True,
        edge_weight=0.3,
        edge_low=-1,
        edge_high=-1,
        edge_auto_sigma=0.33,
        use_rembg=False,
        rembg_dim=0.0,
        rembg_feather=0,
        rembg_erode=0,
        pp_gamma=1.0,
        pp_clip_high=98.0,
    )
    assert result.shape == (16, 16)
    assert result.dtype == np.uint8

# endregion

# region -- Dependency handling --

def test_build_importance_map_returns_valid_range(dummy_gray, monkeypatch):
    class DummyFaceMesh:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def process(self, rgb): return type("Res", (), {"multi_face_landmarks": None})

    monkeypatch.setattr(pp.mp.solutions.face_mesh, "FaceMesh", lambda **_: DummyFaceMesh())

    # warning here stems from the rembg library
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        imp = pp.build_importance_map(dummy_gray, worksize=16)
    assert imp.shape == (16, 16)
    assert imp.dtype == np.float32
    assert np.all((imp >= 0.1) & (imp <= 1.0))

# endregion