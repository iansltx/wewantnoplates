"""Tests for image I/O and target-size computation."""

from wewantnoplates.io_util import compute_target_size, make_output_path


def test_compute_size_landscape():
    # 4:3 landscape -> 1024 wide, 768 tall (multiple of 8).
    assert compute_target_size(960, 720, max_side=1024) == (1024, 768)


def test_compute_size_portrait():
    # 3:4 portrait -> 768 wide, 1024 tall.
    assert compute_target_size(720, 960, max_side=1024) == (768, 1024)


def test_compute_size_uses_config_default():
    from wewantnoplates import config

    assert compute_target_size(1000, 1000) == (config.MAX_SIDE_PIXELS, config.MAX_SIDE_PIXELS)


def test_compute_size_rounds_to_multiple_of_8():
    # 2000:1100 with max_side 1024 -> 1024 wide, tall rounded to multiple of 8.
    w, h = compute_target_size(2000, 1100, max_side=1024)
    assert w % 8 == 0 and h % 8 == 0
    assert w == 1024


def test_compute_size_minimum():
    assert compute_target_size(10, 10, max_side=16) == (16, 16)


def test_make_output_path_unique(tmp_path):
    p1 = make_output_path(extension=".png", out_dir=str(tmp_path))
    assert p1.suffix == ".png"
    assert p1.parent == tmp_path
