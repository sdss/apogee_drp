"""Unit tests for pure ap1dvisit translation helpers."""

import numpy as np

from apogee_drp.apred.visit.driver import (
    sanitize_frame,
    select_visit_objects,
)


def test_sanitize_frame_marks_bad_pixels_once():
    frame = []
    for _ in range(3):
        frame.append(
            {
                "flux": np.array([[1.0, np.nan, 3.0]], dtype=float),
                "error": np.array([[1.0, 2.0, 0.0]], dtype=float),
                "mask": np.zeros((1, 3), dtype=np.int32),
            }
        )

    def arrays(data, chip):
        item = data[chip]
        return item["flux"], item["error"], item["mask"]

    assert sanitize_frame(frame, chip_arrays=arrays, bad_error=999.0) == 6
    for chip in frame:
        np.testing.assert_array_equal(chip["flux"], [[1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(chip["error"], [[1.0, 999.0, 999.0]])
        np.testing.assert_array_equal(chip["mask"], [[0, 1, 1]])


def _fiber_table():
    dtype = [
        ("spectrographid", "i4"),
        ("holetype", "U10"),
        ("objtype", "U10"),
        ("tmass_style", "U20"),
        ("assigned", "i4"),
        ("on_target", "i4"),
        ("valid", "i4"),
    ]
    rows = np.zeros(5, dtype=dtype)
    rows["spectrographid"] = 2
    rows["holetype"] = "OBJECT"
    rows["objtype"] = ["STAR", "HOT_STD", "SKY", "STAR", "STAR"]
    rows["tmass_style"] = ["A", "B", "C", "2MNone", "E"]
    rows["assigned"] = [1, 1, 1, 1, 0]
    rows["on_target"] = 1
    rows["valid"] = 1
    return rows


def test_select_visit_objects_plate_and_fps_rules():
    rows = _fiber_table()
    np.testing.assert_array_equal(select_visit_objects(rows, fps=False), [0, 1, 4])
    np.testing.assert_array_equal(select_visit_objects(rows, fps=True), [0, 1])
