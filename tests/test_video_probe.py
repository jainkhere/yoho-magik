import numpy as np

from reel_curator.frame_sampler import _rotate_frame
from reel_curator.video_probe import _display_dimensions, _rotation_degrees


def test_rotation_metadata_swaps_display_dimensions() -> None:
    stream = {
        "width": 1920,
        "height": 1080,
        "side_data_list": [{"rotation": -90}],
    }

    rotation = _rotation_degrees(stream)

    assert rotation == 270
    assert _display_dimensions(1920, 1080, rotation) == (1080, 1920)


def test_negative_iphone_rotation_rotates_frame_to_portrait() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    rotated = _rotate_frame(frame, 270)

    assert rotated.shape[:2] == (1920, 1080)


def test_already_upright_opencv_frame_is_not_rotated_again() -> None:
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    rotated = _rotate_frame(frame, 270, display_size=(1080, 1920))

    assert rotated.shape[:2] == (1920, 1080)
