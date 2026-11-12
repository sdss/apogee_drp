import numpy as np

from apogee_drp.apred.visit.backend import NativeVisitBackendMixin


class Backend(NativeVisitBackendMixin):
    pass


def _frame():
    frame = {"shift": {"shiftfit": np.zeros(2)}}
    for letter in "abc":
        frame[f"chip{letter}"] = {
            "header": {},
            "flux": np.ones((2, 5)),
            "err": np.full((2, 5), 2.0),
            "mask": np.zeros((2, 5), dtype=np.int32),
            "wavelength": np.tile(np.linspace(15000, 15100, 5), (2, 1)),
            "wcoef": np.zeros((2, 14)),
        }
    return frame


def test_native_backend_frame_compatibility_methods():
    backend = Backend()
    frame = _frame()
    backend.add_header(frame, "TEST", 12)
    backend.add_history(frame, "translated")
    assert backend.header_value(frame, "TEST") == 12
    assert frame["chipb"]["header"]["HISTORY"] == ["translated"]
    flux, error, mask = backend.chip_pixel_arrays(frame, 1)
    assert flux.shape == error.shape == mask.shape == (2, 5)
    assert backend.shift_result(frame)["shiftfit"].tolist() == [0.0, 0.0]


def test_native_backend_combination_routes_object_types():
    backend = Backend()
    frame = _frame()
    plugmap = {
        "fiberdata": {
            "fiberid": np.array([300, 299]),
            "objtype": np.array(["STAR", "SKY"]),
        }
    }
    output, pairs = backend.combine_dithers(
        [frame], [{"index": 0}], plugmap=plugmap, nodither=True
    )
    assert pairs is None
    assert output is not frame
    np.testing.assert_array_equal(output["chipa"]["flux"], 1)


def test_existing_python_wavelength_calibration_is_consumed():
    backend = Backend()
    frame = _frame()
    assert backend.wavelength_calibrate(frame) is frame
    frame["chipa"]["wavelength"] = np.ones((2, 4))
    try:
        backend.wavelength_calibrate(frame)
    except ValueError as error:
        assert "wavelength array" in str(error)
    else:
        raise AssertionError("expected malformed wavelength rejection")
