"""Fast unit tests for shared calibration locking helpers."""

import numpy as np
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from apogee_drp.apred.cal import utils as cal_utils


def calibration_frame(flux, err=None, mask=None, header=None):
    flux = np.asarray(flux)
    return {
        "flux": flux,
        "err": np.ones_like(flux, dtype=float) if err is None else np.asarray(err),
        "mask": np.zeros_like(flux, dtype=np.uint16) if mask is None else np.asarray(mask),
        "header": {} if header is None else header,
    }


class TestAverageCalibrationFrames:
    def test_averages_flux_and_uses_rms_input_error(self):
        frames = [
            calibration_frame([[2.0, 4.0]], err=[[2.0, 3.0]]),
            calibration_frame([[4.0, 8.0]], err=[[2.0, 4.0]]),
        ]
        result = cal_utils.average_calibration_frames(frames)
        assert np.allclose(result["flux"], [[3.0, 6.0]])
        assert np.allclose(
            result["err"], [[2.0, 5.0 / np.sqrt(2.0)]])
        assert np.array_equal(result["mask"], [[0, 0]])

    def test_excludes_masked_and_nonfinite_flux(self):
        frames = [
            calibration_frame(
                [[2.0, np.nan, 3.0]], err=[[1.0, 1.0, 1.0]],
                mask=[[0, 0, 1]]),
            calibration_frame(
                [[4.0, 8.0, np.inf]], err=[[1.0, 2.0, 1.0]]),
        ]
        result = cal_utils.average_calibration_frames(frames)
        assert np.allclose(result["flux"][:, :2], [[3.0, 8.0]])
        assert np.isnan(result["flux"][0, 2])
        assert np.array_equal(result["mask"], [[0, 0, 1]])

    def test_does_not_modify_inputs_and_copies_header(self):
        header = {"TEST": 1}
        frame = calibration_frame([[1.0]], header=header)
        original = frame["flux"].copy()
        result = cal_utils.average_calibration_frames([frame])
        assert np.array_equal(frame["flux"], original)
        assert result["header"] == header
        assert result["header"] is not header

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="at least one"):
            cal_utils.average_calibration_frames([])

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="matching shapes"):
            cal_utils.average_calibration_frames([
                calibration_frame(np.ones((2, 2))),
                calibration_frame(np.ones((3, 2))),
            ])


@pytest.fixture
def mock_lock(monkeypatch):
    mocked = MagicMock()
    monkeypatch.setattr(cal_utils.lock, "lock", mocked)
    return mocked


@pytest.fixture
def product_load(tmp_path):
    filenames = [
        str(tmp_path / "cal" / f"apDetector-{chip}-12345678.fits")
        for chip in ("a", "b", "c")
    ]
    load = MagicMock()
    load.product_files.return_value = filenames
    load.product_exists.return_value = False
    return load, filenames


class TestCalibrationLock:
    def test_acquires_and_clears_lock(self, tmp_path, mock_lock):
        filename = tmp_path / "product.fits"

        with cal_utils.calibration_lock(
            filename,
            waittime=37,
            unlock=True,
        ):
            assert mock_lock.call_args_list == [
                call(str(filename), waittime=37, unlock=True),
                call(str(filename), lock=True),
            ]

        assert mock_lock.call_args_list == [
            call(str(filename), waittime=37, unlock=True),
            call(str(filename), lock=True),
            call(str(filename), clear=True),
        ]

    def test_yields_none(self, tmp_path, mock_lock):
        with cal_utils.calibration_lock(tmp_path / "product.fits") as value:
            assert value is None

    def test_accepts_string_filename(self, mock_lock):
        with cal_utils.calibration_lock("/cal/product.fits"):
            pass

        assert mock_lock.call_args_list == [
            call("/cal/product.fits", waittime=10, unlock=False),
            call("/cal/product.fits", lock=True),
            call("/cal/product.fits", clear=True),
        ]

    def test_clears_lock_after_body_exception(self, tmp_path, mock_lock):
        filename = tmp_path / "product.fits"

        with pytest.raises(RuntimeError, match="build failed"):
            with cal_utils.calibration_lock(filename):
                raise RuntimeError("build failed")

        assert mock_lock.call_args_list[-1] == call(
            str(filename), clear=True
        )

    def test_does_not_clear_unacquired_lock(self, tmp_path, mock_lock):
        filename = tmp_path / "product.fits"
        mock_lock.side_effect = [None, RuntimeError("cannot acquire")]

        with pytest.raises(RuntimeError, match="cannot acquire"):
            with cal_utils.calibration_lock(filename):
                pass

        assert mock_lock.call_args_list == [
            call(str(filename), waittime=10, unlock=False),
            call(str(filename), lock=True),
        ]


class TestFileBuildLock:
    def test_existing_nonempty_file_uses_fast_path(self, tmp_path, mock_lock):
        filename = tmp_path / "measurements.dat"
        filename.write_bytes(b"measurements")

        with cal_utils.file_build_lock(filename) as build:
            assert build is False

        mock_lock.assert_not_called()
        assert filename.read_bytes() == b"measurements"

    def test_fast_path_reports_existing_file(
        self, tmp_path, mock_lock, capsys
    ):
        filename = tmp_path / "measurements.dat"
        filename.write_bytes(b"measurements")

        with cal_utils.file_build_lock(filename, verbose=True) as build:
            assert build is False

        assert f"File {filename} already exists" in capsys.readouterr().out

    def test_missing_file_requests_build_and_creates_parent(
        self, tmp_path, mock_lock
    ):
        filename = tmp_path / "new" / "measurements.dat"

        with cal_utils.file_build_lock(filename) as build:
            assert build is True
            assert filename.parent.is_dir()
            assert not filename.exists()

        assert mock_lock.call_args_list[-1] == call(
            str(filename), clear=True
        )

    def test_empty_file_is_removed_before_build(self, tmp_path, mock_lock):
        filename = tmp_path / "measurements.dat"
        filename.touch()

        with cal_utils.file_build_lock(filename) as build:
            assert build is True
            assert not filename.exists()

    def test_clobber_removes_complete_file(self, tmp_path, mock_lock):
        filename = tmp_path / "measurements.dat"
        filename.write_bytes(b"old")

        with cal_utils.file_build_lock(filename, clobber=True) as build:
            assert build is True
            assert not filename.exists()

    def test_removes_broken_symlink_before_build(self, tmp_path, mock_lock):
        target = tmp_path / "missing-target"
        filename = tmp_path / "measurements.dat"
        filename.symlink_to(target)

        with cal_utils.file_build_lock(filename) as build:
            assert build is True
            assert not filename.is_symlink()

    def test_file_created_while_waiting_is_reused(
        self, tmp_path, monkeypatch
    ):
        filename = tmp_path / "measurements.dat"

        @contextmanager
        def create_while_waiting(*args, **kwargs):
            filename.write_bytes(b"created elsewhere")
            yield

        monkeypatch.setattr(
            cal_utils, "calibration_lock", create_while_waiting
        )

        with cal_utils.file_build_lock(filename) as build:
            assert build is False

        assert filename.read_bytes() == b"created elsewhere"

    def test_race_path_reports_existing_file(
        self, tmp_path, monkeypatch, capsys
    ):
        filename = tmp_path / "measurements.dat"

        @contextmanager
        def create_while_waiting(*args, **kwargs):
            filename.write_bytes(b"created elsewhere")
            yield

        monkeypatch.setattr(
            cal_utils, "calibration_lock", create_while_waiting
        )

        with cal_utils.file_build_lock(filename, verbose=True):
            pass

        assert f"File {filename} already exists" in capsys.readouterr().out

    def test_forwards_lock_options(self, tmp_path, mock_lock):
        filename = tmp_path / "measurements.dat"

        with cal_utils.file_build_lock(
            filename,
            waittime=42,
            unlock=True,
        ):
            pass

        assert mock_lock.call_args_list[:2] == [
            call(str(filename), waittime=42, unlock=True),
            call(str(filename), lock=True),
        ]

    def test_clears_lock_after_body_exception(self, tmp_path, mock_lock):
        filename = tmp_path / "measurements.dat"

        with pytest.raises(RuntimeError, match="measurement failed"):
            with cal_utils.file_build_lock(filename):
                raise RuntimeError("measurement failed")

        assert mock_lock.call_args_list[-1] == call(
            str(filename), clear=True
        )


class TestProductBuildLock:
    def test_existing_product_uses_fast_path(
        self, product_load, mock_lock
    ):
        load, filenames = product_load
        load.product_exists.return_value = True

        with cal_utils.product_build_lock(
            load, "detector", 12345678
        ) as result:
            assert result == (False, filenames)

        mock_lock.assert_not_called()
        load.product_delete.assert_not_called()

    def test_fast_path_reports_existing_product(
        self, product_load, mock_lock, capsys
    ):
        load, _ = product_load
        load.product_exists.return_value = True

        with cal_utils.product_build_lock(
            load,
            "detector",
            12345678,
            verbose=True,
        ):
            pass

        assert (
            "detector product 12345678 already exists"
            in capsys.readouterr().out
        )

    def test_missing_product_requests_build_and_deletes_leftovers(
        self, product_load, mock_lock
    ):
        load, filenames = product_load

        with cal_utils.product_build_lock(
            load, "detector", 12345678
        ) as result:
            assert result == (True, filenames)
            assert Path(filenames[0]).parent.is_dir()

        load.product_delete.assert_called_once_with(
            "detector", 12345678, verbose=False
        )

    def test_product_created_while_waiting_is_reused(
        self, product_load, mock_lock
    ):
        load, filenames = product_load
        load.product_exists.side_effect = [False, True]

        with cal_utils.product_build_lock(
            load, "detector", 12345678
        ) as result:
            assert result == (False, filenames)

        load.product_delete.assert_not_called()
        assert mock_lock.call_args_list[-1] == call(
            filenames[0], clear=True
        )

    def test_race_path_reports_existing_product(
        self, product_load, mock_lock, capsys
    ):
        load, _ = product_load
        load.product_exists.side_effect = [False, True]

        with cal_utils.product_build_lock(
            load,
            "detector",
            12345678,
            verbose=True,
        ):
            pass

        assert (
            "detector product 12345678 already exists"
            in capsys.readouterr().out
        )

    def test_clobber_forces_build(self, product_load, mock_lock):
        load, filenames = product_load
        load.product_exists.return_value = True

        with cal_utils.product_build_lock(
            load,
            "detector",
            12345678,
            clobber=True,
        ) as result:
            assert result == (True, filenames)

        load.product_delete.assert_called_once_with(
            "detector", 12345678, verbose=False
        )

    def test_forwards_verbose_to_product_delete(
        self, product_load, mock_lock
    ):
        load, _ = product_load

        with cal_utils.product_build_lock(
            load,
            "detector",
            12345678,
            verbose=True,
        ):
            pass

        load.product_delete.assert_called_once_with(
            "detector", 12345678, verbose=True
        )

    def test_forwards_lock_options(self, product_load, mock_lock):
        load, filenames = product_load

        with cal_utils.product_build_lock(
            load,
            "detector",
            12345678,
            waittime=91,
            unlock=True,
        ):
            pass

        assert mock_lock.call_args_list[:2] == [
            call(filenames[0], waittime=91, unlock=True),
            call(filenames[0], lock=True),
        ]

    def test_clears_lock_after_body_exception(
        self, product_load, mock_lock
    ):
        load, filenames = product_load

        with pytest.raises(RuntimeError, match="calibration failed"):
            with cal_utils.product_build_lock(
                load, "detector", 12345678
            ):
                raise RuntimeError("calibration failed")

        assert mock_lock.call_args_list[-1] == call(
            filenames[0], clear=True
        )

    def test_clears_lock_when_product_delete_fails(
        self, product_load, mock_lock
    ):
        load, filenames = product_load
        load.product_delete.side_effect = RuntimeError("delete failed")

        with pytest.raises(RuntimeError, match="delete failed"):
            with cal_utils.product_build_lock(
                load, "detector", 12345678
            ):
                pass

        assert mock_lock.call_args_list[-1] == call(
            filenames[0], clear=True
        )

    def test_uses_registry_filenames(self, product_load, mock_lock):
        load, filenames = product_load

        with cal_utils.product_build_lock(
            load, "detector", "12345678-87654321"
        ) as (_, yielded):
            assert yielded is filenames

        load.product_files.assert_called_once_with(
            "detector", "12345678-87654321"
        )


def test_nan_uniform_filter_ignores_bad_pixels():
    image = np.ones((9, 9))
    image[4, 4] = np.nan
    smoothed = cal_utils.nan_uniform_filter(image, 3)
    np.testing.assert_allclose(smoothed, 1)


def test_safe_divide_marks_invalid_values():
    result = cal_utils.safe_divide([2, 2, np.nan], [2, 0, 1])
    assert result[0] == 1
    assert np.isnan(result[1:]).all()
