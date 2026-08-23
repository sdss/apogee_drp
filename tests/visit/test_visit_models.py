import numpy as np
import pytest

from apogee_drp.apred.visit.models import ChipFrame, VisitFrame


def chip(value=1.0, nfiber=2, npix=4):
    shape = (nfiber, npix)
    return ChipFrame(np.full(shape, value), np.ones(shape),
                     np.zeros(shape, dtype=np.int16))


def test_visit_frame_accesses_chips_by_name_and_index():
    frame = VisitFrame(chip(1), chip(2), chip(3))
    assert frame.chip("b") is frame.chipb
    assert frame[2] is frame.chipc
    assert frame["chipa"] is frame.chipa


def test_visit_frame_preserves_extra_metadata():
    frame = VisitFrame(chip(), chip(), chip())
    frame["plugmap"] = {"name": "test"}
    assert frame.get("plugmap") == {"name": "test"}
    assert frame.get("missing", 12) == 12


def test_visit_frame_converts_legacy_mapping():
    data = {f"chip{name}": {"flux": np.ones((2, 3)),
            "error": np.ones((2, 3)), "mask": np.zeros((2, 3), dtype=int)}
            for name in "abc"}
    frame = VisitFrame.from_mapping(data)
    assert isinstance(frame.chipa, ChipFrame)
    assert frame.chipa.error is frame.chipa.err


def test_chip_frame_rejects_mismatched_arrays():
    value = chip()
    value.err = np.ones((3, 4))
    with pytest.raises(ValueError, match="identical shapes"):
        value.validate()


def test_visit_frame_requires_common_fiber_count():
    frame = VisitFrame(chip(nfiber=2), chip(nfiber=3), chip(nfiber=2))
    with pytest.raises(ValueError, match="same number of fibers"):
        frame.validate()
