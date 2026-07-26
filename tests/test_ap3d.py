"""Extensive tests for the APOGEE 3-D to 2-D reduction.

The suite has three levels:

* ordinary tests are small, fast unit tests;
* tests marked ``slow`` allocate full 2048-pixel detector arrays;
* tests marked ``realdata`` use APOGEE files supplied through environment
  variables.

Run the fast suite with::

    pytest -q -m "not slow and not realdata" test_ap3d.py

Run all synthetic tests with::

    pytest -q -m "not realdata" test_ap3d.py

See ``test_real_apogee_reduction`` for the environment variables used by the
real-data test.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

try:
    from apogee_drp.apred import ap3d
except ImportError:
    # This fallback makes the downloaded test file usable beside ap3d.py.
    import ap3d


def _linear_cube(nread=6, ny=2, nx=3, intercept=100.0, signal=None):
    """Return a small exactly linear test ramp."""

    if signal is None:
        signal = np.arange(1, ny * nx + 1, dtype=np.float32).reshape(ny, nx)
    return np.stack(
        [intercept + read * signal for read in range(nread)]
    ).astype(np.float32)


def _process_result(shape=(3, 4), persistence=False):
    """Return a small ProcessResult suitable for FITS writer tests."""

    pmodel = np.ones(shape, dtype=np.float32) if persistence else None
    return ap3d.ProcessResult(
        flux=np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        error=np.full(shape, 2.5, dtype=np.float32),
        mask=np.zeros(shape, dtype=np.uint16),
        header=fits.Header({"BUNIT": "ADU", "CHIP": "c"}),
        persistence_model=pmodel,
    )


class TestDataContainers:
    """Tests for reduction result and event containers."""

    def test_cosmic_ray_defaults(self):
        """CosmicRay supplies neutral diagnostic defaults."""

        event = ap3d.CosmicRay(x=12)
        assert event.x == 12
        assert event.y == -1
        assert event.read == 0
        assert not event.fixed
        assert event.fix_error == 0.0

    def test_process_result_defaults(self):
        """Optional ProcessResult diagnostics default to empty values."""

        result = _process_result()
        assert result.cosmic_rays == []
        assert result.saturation is None
        assert result.fixed_cube is None
        assert result.read_mask is None
        assert result.global_variability == -1.0


class TestSmallNumericalHelpers:
    """Fast unit tests for private numerical building blocks."""

    def test_rolling_nanmedian_width_one_returns_copy(self):
        """A width-one rolling median preserves values but returns a copy."""

        values = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        result = ap3d._rolling_nanmedian(values, 1)
        np.testing.assert_equal(result, values)
        assert result is not values

    def test_rolling_nanmedian_edge_padding_and_nans(self):
        """The rolling median ignores NaNs and repeats edge samples."""

        values = np.array([[1.0, np.nan, 9.0, 3.0]], dtype=np.float32)
        result = ap3d._rolling_nanmedian(values, 3)
        np.testing.assert_allclose(result, [[1.0, 5.0, 6.0, 3.0]])

    def test_expand_quadrants(self):
        """Four values expand into four 512-column detector outputs."""

        result = ap3d._expand_quadrants([1, 2, 3, 4], (2, 2048))
        assert result.shape == (2, 2048)
        for quadrant, value in enumerate((1, 2, 3, 4)):
            np.testing.assert_array_equal(
                result[:, quadrant * 512 : (quadrant + 1) * 512], value
            )

    @pytest.mark.parametrize(
        "values,shape,message",
        [
            ([1, 2, 3], (2, 2048), "four detector-output"),
            ([1, 2, 3, 4], (2, 1024), "requires nx=2048"),
        ],
    )
    def test_expand_quadrants_rejects_invalid_input(self, values, shape, message):
        """Quadrant expansion validates both value count and geometry."""

        with pytest.raises(ValueError, match=message):
            ap3d._expand_quadrants(values, shape)

    def test_reference_subtract_image_orientation(self):
        """Alternating outputs use the horizontally reversed reference."""

        reference = np.arange(512, dtype=np.float32)[None, :]
        image = np.zeros((1, 2048), dtype=np.float32)
        ap3d._reference_subtract_image(image, reference)
        np.testing.assert_array_equal(image[:, :512], -reference)
        np.testing.assert_array_equal(image[:, 512:1024], -reference[:, ::-1])
        np.testing.assert_array_equal(image[:, 1024:1536], -reference)
        np.testing.assert_array_equal(image[:, 1536:], -reference[:, ::-1])

    def test_median_filter_removes_isolated_outlier(self):
        """The two-dimensional median filter suppresses a central spike."""

        image = np.ones((5, 5), dtype=np.float32)
        image[2, 2] = 100
        result = ap3d._median_filter_2d(image, 3)
        assert result[2, 2] == 1


class TestBadReadHandling:
    """Tests for rejected-read interpolation."""

    def test_interpolate_interior_read(self):
        """An interior rejected read is linearly interpolated."""

        cube = np.array([0.0, 10.0, 999.0, 30.0, 40.0])[:, None, None]
        result = ap3d._interpolate_bad_reads(
            cube, np.array([False, False, True, False, False])
        )
        assert result[2, 0, 0] == 20.0
        assert cube[2, 0, 0] == 999.0

    @pytest.mark.parametrize(
        "bad,expected",
        [
            ([True, False, False, False], 0.0),
            ([False, False, False, True], 30.0),
        ],
    )
    def test_interpolate_end_read_extrapolates(self, bad, expected):
        """A rejected end read follows the line through two good reads."""

        cube = np.array([0.0, 10.0, 20.0, 30.0])[:, None, None]
        result = ap3d._interpolate_bad_reads(cube, np.array(bad))
        index = 0 if bad[0] else -1
        assert result[index, 0, 0] == expected

    def test_interpolate_requires_two_good_reads(self):
        """Interpolation fails when fewer than two good reads remain."""

        cube = np.arange(3, dtype=np.float32)[:, None, None]
        with pytest.raises(ValueError, match="At least two good reads"):
            ap3d._interpolate_bad_reads(cube, np.array([True, True, False]))


class TestLinearity:
    """Tests for detector linearity polynomial application."""

    def test_pixel_linearity_coefficients(self):
        """Per-pixel quadratic coefficients are evaluated correctly."""

        cube = np.array([[[2.0, 3.0]]], dtype=np.float32)
        coefficients = np.zeros((1, 2, 3), dtype=np.float32)
        coefficients[..., 0] = 1.0
        coefficients[..., 1] = 2.0
        coefficients[..., 2] = 3.0
        result = ap3d._apply_linearity(cube, coefficients)
        np.testing.assert_allclose(result, 1 + 2 * cube + 3 * cube**2)

    def test_quadrant_linearity_coefficients(self):
        """Each detector output receives its own polynomial."""

        cube = np.ones((1, 1, 2048), dtype=np.float32) * 2
        coefficients = np.array(
            [[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.float32
        )
        result = ap3d._apply_linearity(cube, coefficients)
        for quadrant, scale in enumerate((1, 2, 3, 4)):
            np.testing.assert_allclose(
                result[:, :, quadrant * 512 : (quadrant + 1) * 512],
                2 * scale,
            )

    def test_transposed_quadrant_coefficients(self):
        """Astropy-style transposed output coefficients are accepted."""

        cube = np.ones((1, 1, 2048), dtype=np.float32) * 3
        coefficients = np.array(
            [[0, 0, 0, 0], [1, 2, 3, 4]], dtype=np.float32
        )
        result = ap3d._apply_linearity(cube, coefficients)
        assert result[0, 0, 0] == 3
        assert result[0, 0, 600] == 6
        assert result[0, 0, 1200] == 9
        assert result[0, 0, 1800] == 12

    def test_invalid_linearity_shape(self):
        """Unrecognized coefficient shapes raise an informative error."""

        with pytest.raises(ValueError, match="Linearity coefficients"):
            ap3d._apply_linearity(
                np.ones((2, 2, 2), dtype=np.float32),
                np.ones((5, 5), dtype=np.float32),
            )


class TestCosmicRays:
    """Tests for read-difference cosmic-ray detection."""

    def test_detect_and_fix_positive_event(self):
        """A single positive outlier is detected and replaced."""

        dcounts = np.full((3, 15), 100.0, dtype=np.float32)
        dcounts[1, 7] = 1000.0
        saturation = np.zeros((3, 3), dtype=np.int32)
        fixed, median, variability, events = ap3d.detect_and_fix_cosmic_rays(
            dcounts, saturation, noise=5.0
        )
        assert len(events) == 1
        event = events[0]
        assert event.x == 1
        assert event.read == 8
        assert event.fixed
        assert event.counts > 0
        assert event.nsigma > 10
        assert event.fix_error > 0
        assert fixed[1, 7] == median[1] == 100.0
        assert variability.shape == (3,)

    def test_detect_without_fix_preserves_data(self):
        """Detection-only mode reports an event without changing samples."""

        dcounts = np.full((1, 15), 50.0, dtype=np.float32)
        dcounts[0, 6] = 500
        fixed, _, _, events = ap3d.detect_and_fix_cosmic_rays(
            dcounts, np.zeros((1, 3), dtype=int), noise=2.0, fix=False
        )
        np.testing.assert_array_equal(fixed, dcounts)
        assert len(events) == 1
        assert not events[0].fixed
        assert events[0].fix_error == 0.0

    def test_only_read_filters_events(self):
        """The optional read constraint excludes distant events."""

        dcounts = np.full((1, 15), 50.0, dtype=np.float32)
        dcounts[0, 10] = 500
        _, _, _, events = ap3d.detect_and_fix_cosmic_rays(
            dcounts,
            np.zeros((1, 3), dtype=int),
            noise=2.0,
            only_read=2,
        )
        assert events == []

    def test_constant_differences_have_no_events(self):
        """A constant ramp produces no false cosmic-ray detections."""

        dcounts = np.full((5, 20), 100.0, dtype=np.float32)
        fixed, median, variability, events = ap3d.detect_and_fix_cosmic_rays(
            dcounts, np.zeros((5, 3), dtype=int), noise=5.0
        )
        np.testing.assert_array_equal(fixed, dcounts)
        np.testing.assert_allclose(median, 100.0)
        np.testing.assert_allclose(variability, 0.0)
        assert events == []


class TestFowlerSampling:
    """Tests for standalone Fowler sampling."""

    def test_requested_samples(self):
        """The means of two reads at each end form the Fowler image."""

        signal = np.array([[2.0, 3.0]], dtype=np.float32)
        cube = _linear_cube(nread=6, ny=1, nx=2, signal=signal)
        image, noise, used = ap3d.fowler_sampling(
            cube, np.arange(6), np.full(signal.shape, 10.0), nfowler=2
        )
        np.testing.assert_allclose(image, 4.0 * signal)
        np.testing.assert_allclose(noise, 10.0)
        assert used == 2

    def test_bad_read_indices_are_skipped(self):
        """Fowler samples are selected from the supplied good-read list."""

        cube = _linear_cube(nread=7, ny=1, nx=1, signal=np.array([[5.0]]))
        image, _, used = ap3d.fowler_sampling(
            cube, np.array([0, 2, 4, 6]), 10.0, nfowler=1
        )
        np.testing.assert_allclose(image, [[30.0]])
        assert used == 1

    def test_nfowler_is_limited_by_good_reads(self):
        """Each sample is limited to half the available good reads."""

        cube = np.arange(5, dtype=np.float32)[:, None, None]
        image, noise, used = ap3d.fowler_sampling(
            cube, np.arange(5), 4.0, nfowler=10
        )
        np.testing.assert_allclose(image, [[3.0]])
        np.testing.assert_allclose(noise, 4.0)
        assert used == 2

    @pytest.mark.parametrize(
        "reads,nfowler,message",
        [
            ([0], 1, "at least two good reads"),
            ([0, 1], 0, "at least one"),
            ([0, 4], 1, "outside the ramp"),
        ],
    )
    def test_invalid_inputs(self, reads, nfowler, message):
        """Fowler sampling rejects invalid read selections and sample sizes."""

        cube = np.zeros((4, 1, 1), dtype=np.float32)
        with pytest.raises(ValueError, match=message):
            ap3d.fowler_sampling(cube, reads, 1.0, nfowler=nfowler)

    def test_requires_three_dimensional_cube(self):
        """Fowler sampling requires the read axis plus two image axes."""

        with pytest.raises(ValueError, match="cube must have shape"):
            ap3d.fowler_sampling(np.zeros((3, 2)), [0, 1], 1.0)


class TestUpTheRampSampling:
    """Tests for standalone up-the-ramp sampling."""

    def test_linear_ramp(self):
        """The fitted slope recovers an exactly linear ramp."""

        signal = np.array([[2.0, 3.0]], dtype=np.float32)
        cube = _linear_cube(nread=5, ny=1, nx=2, signal=signal)
        image, noise = ap3d.up_the_ramp_sampling(
            cube,
            np.arange(5),
            np.full(signal.shape, 10.0),
            np.full(signal.shape, 2.0),
            science_nx=2,
        )
        np.testing.assert_allclose(image, 4.0 * signal)
        assert noise.shape == signal.shape
        assert np.all(np.isfinite(noise))
        assert np.all(noise > 0)

    def test_nonconsecutive_good_reads(self):
        """The fit uses original read indices when rejected reads leave gaps."""

        signal = np.array([[7.0]], dtype=np.float32)
        cube = _linear_cube(nread=7, ny=1, nx=1, signal=signal)
        image, _ = ap3d.up_the_ramp_sampling(
            cube, [0, 2, 5, 6], 10.0, 2.0, science_nx=1
        )
        # The IDL convention scales the slope by Ngood-1, not the time span.
        np.testing.assert_allclose(image, [[21.0]], rtol=1e-6)

    def test_reference_columns_retained_only_in_image(self):
        """Reference columns remain in flux but not in the noise image."""

        cube = _linear_cube(nread=4, ny=2, nx=5)
        image, noise = ap3d.up_the_ramp_sampling(
            cube, np.arange(4), np.ones((2, 3)), np.ones((2, 3)), science_nx=3
        )
        assert image.shape == (2, 5)
        assert noise.shape == (2, 3)

    def test_noise_matches_formula(self):
        """The returned UTR uncertainty follows the Rauscher expression."""

        cube = _linear_cube(
            nread=5, ny=1, nx=1, signal=np.array([[4.0]], dtype=np.float32)
        )
        image, noise = ap3d.up_the_ramp_sampling(
            cube, np.arange(5), 10.0, 2.0, science_nx=1
        )
        n = 5.0
        expected = np.sqrt(
            12 * (n - 1) / (n * (n + 1)) * 10.0**2
            + 6 * (n**2 + 1) / (5 * n * (n + 1)) * image * 2.0
        ) / 2.0
        np.testing.assert_allclose(noise, expected)

    @pytest.mark.parametrize(
        "reads,science_nx,message",
        [
            ([0], 1, "at least two good reads"),
            ([0, 5], 1, "outside the ramp"),
            ([0, 1], 0, "science_nx"),
            ([0, 1], 4, "science_nx"),
        ],
    )
    def test_invalid_inputs(self, reads, science_nx, message):
        """UTR sampling validates read indices and science geometry."""

        cube = np.zeros((4, 1, 3), dtype=np.float32)
        with pytest.raises(ValueError, match=message):
            ap3d.up_the_ramp_sampling(
                cube, reads, 1.0, 1.0, science_nx=science_nx
            )


class TestHeader:
    """Tests for output-header bookkeeping."""

    @pytest.mark.parametrize(
        "up_the_ramp,nfowler,history",
        [
            (False, 3, "Fowler sampling, Nfowler=3"),
            (True, None, "Up-the-ramp sampling"),
        ],
    )
    def test_sampling_history(self, up_the_ramp, nfowler, history):
        """The selected sampling method is recorded in HISTORY."""

        header = fits.Header({"NFRAMES": 4})
        ap3d._update_header(
            header,
            nread=4,
            gain=2.0,
            readnoise=5.0,
            up_the_ramp=up_the_ramp,
            nfowler=nfowler,
            global_variability=0.1,
            output_electrons=False,
        )
        assert history in list(header["HISTORY"])
        assert header["GAIN"] == 2.0
        assert header["RDNOISE"] == 5.0
        assert header["BUNIT"] == "ADU"

    def test_header_updates_exposure_and_midpoint(self):
        """A mismatched read count updates exposure time and midpoint."""

        header = fits.Header(
            {
                "NFRAMES": 99,
                "DATE-OBS": "2025-01-01T00:00:00.000",
                "CHECKSUM": "old",
                "DATASUM": "old",
            }
        )
        ap3d._update_header(
            header,
            nread=4,
            gain=2.0,
            readnoise=5.0,
            up_the_ramp=False,
            nfowler=1,
            global_variability=-1.0,
            output_electrons=True,
        )
        assert header["EXPTIME"] == pytest.approx(4 * 10.647)
        assert "UT-MID" in header
        assert "JD-MID" in header
        assert header["BUNIT"] == "electron"
        assert "CHECKSUM" not in header
        assert "DATASUM" not in header


class TestRampIO:
    """Tests for decompressed ramp FITS input."""

    def test_primary_three_dimensional_ramp(self, tmp_path):
        """A 3-D primary image is read directly and can be truncated."""

        cube = np.arange(4 * 2 * 3, dtype=np.uint16).reshape(4, 2, 3)
        filename = tmp_path / "ramp.fits"
        fits.PrimaryHDU(cube, header=fits.Header({"CHIP": "b"})).writeto(filename)
        result, header = ap3d.read_ramp(filename, max_read=3)
        np.testing.assert_array_equal(result, cube[:3])
        assert header["CHIP"] == "b"

    def test_unsigned_scaled_primary_ramp(self, tmp_path):
        """Unsigned FITS scaling is read without strict-memory-map failure."""

        cube = np.array([[[0, 65535]], [[1, 50000]]], dtype=np.uint16)
        filename = tmp_path / "unsigned.fits"
        fits.PrimaryHDU(cube).writeto(filename)
        result, _ = ap3d.read_ramp(filename)
        np.testing.assert_array_equal(result, cube)

    def test_image_extension_ramp_and_metadata(self, tmp_path):
        """Separate image extensions are stacked and donate metadata."""

        filename = tmp_path / "extensions.fits"
        hdus = [fits.PrimaryHDU()]
        for read in range(3):
            hdu = fits.ImageHDU(np.full((2, 3), read, dtype=np.int16))
            hdu.header["CHIP"] = "c"
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(filename)
        cube, header = ap3d.read_ramp(filename, max_read=2)
        assert cube.shape == (2, 2, 3)
        np.testing.assert_array_equal(cube[:, 0, 0], [0, 1])
        assert header["CHIP"] == "c"

    def test_rejects_apz(self, tmp_path):
        """The numerical FITS reader rejects compressed APZ input."""

        filename = tmp_path / "test.apz"
        filename.touch()
        with pytest.raises(ValueError, match="requires a decompressed FITS"):
            ap3d.read_ramp(filename)

    def test_rejects_fewer_than_two_extension_reads(self, tmp_path):
        """An extension-based ramp must contain at least two images."""

        filename = tmp_path / "one-read.fits"
        fits.HDUList(
            [fits.PrimaryHDU(), fits.ImageHDU(np.zeros((2, 3)))]
        ).writeto(filename)
        with pytest.raises(ValueError, match="fewer than two"):
            ap3d.read_ramp(filename)


class TestCalibrationIO:
    """Tests for calibration-product loading."""

    def test_detector_and_image_calibrations(self, tmp_path):
        """Detector coefficients and ordinary image calibrations are loaded."""

        detector = tmp_path / "detector.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(np.arange(4, dtype=np.float32) + 10),
                fits.ImageHDU(np.arange(4, dtype=np.float32) + 1),
                fits.ImageHDU(np.arange(12, dtype=np.float32).reshape(3, 4)),
            ]
        ).writeto(detector)
        image_paths = {}
        for name in ("bpm", "flat", "littrow", "persistence_mask"):
            path = tmp_path / f"{name}.fits"
            fits.PrimaryHDU(np.ones((3, 4), dtype=np.float32)).writeto(path)
            image_paths[name] = path
        result = ap3d.read_calibrations(detector=detector, **image_paths)
        assert result["readnoise"].shape == (4,)
        assert result["gain"].shape == (4,)
        assert result["linearity"].shape == (4, 3)
        for name in image_paths:
            assert result[name].shape == (3, 4)

    def test_dark_uses_first_three_dimensional_hdu(self, tmp_path):
        """A 3-D dark ramp is selected without stacking unrelated HDUs."""

        dark = tmp_path / "dark3d.fits"
        expected = np.arange(3 * 2 * 2).reshape(3, 2, 2)
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(expected),
                fits.ImageHDU(np.ones((5, 7))),
            ]
        ).writeto(dark)
        result = ap3d.read_calibrations(dark=dark)
        np.testing.assert_array_equal(result["dark"], expected)

    def test_dark_stacks_matching_two_dimensional_reads(self, tmp_path):
        """Legacy 2-D dark extensions are stacked along the read axis."""

        dark = tmp_path / "dark2d.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(np.zeros((2, 3))),
                fits.ImageHDU(np.ones((2, 3))),
            ]
        ).writeto(dark)
        result = ap3d.read_calibrations(dark=dark)
        assert result["dark"].shape == (2, 2, 3)
        np.testing.assert_array_equal(result["dark"][:, 0, 0], [0, 1])

    def test_dark_rejects_inconsistent_two_dimensional_reads(self, tmp_path):
        """Legacy dark extensions must all have matching dimensions."""

        dark = tmp_path / "bad-dark.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(np.zeros((2, 3))),
                fits.ImageHDU(np.ones((3, 2))),
            ]
        ).writeto(dark)
        with pytest.raises(ValueError, match="inconsistent shapes"):
            ap3d.read_calibrations(dark=dark)


class TestOutputIO:
    """Tests for ap2D FITS output."""

    def test_write_float_product(self, tmp_path):
        """The writer creates FLUX, ERROR, and MASK extensions with units."""

        filename = tmp_path / "ap2d.fits"
        result = _process_result()
        ap3d.write_ap2d(filename, result)
        with fits.open(filename) as hdul:
            assert [hdu.name for hdu in hdul] == [
                "PRIMARY",
                "FLUX",
                "ERROR",
                "MASK",
            ]
            np.testing.assert_array_equal(hdul["FLUX"].data, result.flux)
            np.testing.assert_array_equal(hdul["ERROR"].data, result.error)
            assert hdul["MASK"].data.dtype.kind == "u"
            assert hdul["FLUX"].header["BUNIT"] == "ADU"
            assert hdul["MASK"].header["BUNIT"] == "bitwise"
            assert hdul[0].verify_checksum() == 1

    def test_write_integer_product_and_persistence(self, tmp_path):
        """Integer output rounds arrays and includes persistence correction."""

        filename = tmp_path / "integer.fits"
        result = _process_result(persistence=True)
        result.flux = result.flux + 0.6
        ap3d.write_ap2d(filename, result, integer_output=True)
        with fits.open(filename) as hdul:
            assert hdul["FLUX"].data.dtype.kind == "i"
            np.testing.assert_array_equal(
                hdul["FLUX"].data, np.rint(result.flux).astype(np.int32)
            )
            assert "PERSIST CORRECTION" in hdul

    def test_overwrite_protection(self, tmp_path):
        """Existing products are protected unless overwrite is requested."""

        filename = tmp_path / "existing.fits"
        ap3d.write_ap2d(filename, _process_result())
        with pytest.raises(OSError):
            ap3d.write_ap2d(filename, _process_result())
        ap3d.write_ap2d(filename, _process_result(), overwrite=True)


class TestOrchestration:
    """Fast tests of file-level control flow using monkeypatching."""

    def test_process_cube_rejects_apz(self):
        """process_cube accepts only a decompressed FITS ramp."""

        with pytest.raises(ValueError, match="requires a decompressed FITS"):
            ap3d.process_cube("raw.apz")

    def test_process_cube_passes_data_and_options(self, monkeypatch):
        """process_cube forwards loaded data, header, and reduction options."""

        cube = np.ones((3, 2, 2), dtype=np.float32)
        header = fits.Header({"CHIP": "a"})
        calls = {}

        def fake_read(filename, max_read=None, verbose=False):
            """Record the requested ramp read and return synthetic data."""

            calls["read"] = (Path(filename), max_read, verbose)
            return cube, header

        sentinel = object()

        def fake_process(raw, hdr, **options):
            """Record numerical-processing arguments and return a sentinel."""

            calls["process"] = (raw, hdr, options)
            return sentinel

        monkeypatch.setattr(ap3d, "read_ramp", fake_read)
        monkeypatch.setattr(ap3d, "process_array", fake_process)
        result = ap3d.process_cube(
            "ramp.fits",
            max_read=2,
            verbose=True,
            debug=True,
            nfowler=3,
        )
        assert result is sentinel
        assert calls["read"] == (Path("ramp.fits"), 2, True)
        assert calls["process"][0] is cube
        assert calls["process"][1] is header
        assert calls["process"][2]["nfowler"] == 3
        assert calls["process"][2]["debug"] is True

    def test_process_file_fits_orchestration(self, tmp_path, monkeypatch):
        """FITS orchestration loads calibrations, reduces, and writes."""

        input_file = tmp_path / "ramp.fits"
        input_file.touch()
        output = tmp_path / "ap2d.fits"
        result = _process_result()
        calls = {}

        def fake_calibrations(**paths):
            """Record calibration paths and return one synthetic calibration."""

            calls["calibrations"] = paths
            return {"gain": np.array([2.0])}

        def fake_process(filename, **options):
            """Record file-processing arguments and return a test result."""

            calls["process"] = (Path(filename), options)
            return result

        def fake_write(filename, value, overwrite=False):
            """Record output-writing arguments without touching disk."""

            calls["write"] = (Path(filename), value, overwrite)

        monkeypatch.setattr(ap3d, "read_calibrations", fake_calibrations)
        monkeypatch.setattr(ap3d, "process_cube", fake_process)
        monkeypatch.setattr(ap3d, "write_ap2d", fake_write)
        returned = ap3d.process_file(
            input_file,
            output,
            detector="detector.fits",
            overwrite=True,
            nfowler=2,
        )
        assert returned is result
        assert calls["calibrations"]["detector"] == "detector.fits"
        assert calls["process"][0] == input_file
        assert calls["process"][1]["nfowler"] == 2
        assert calls["write"] == (output, result, True)

    def test_process_file_apz_uses_apzip(self, tmp_path, monkeypatch):
        """APZ orchestration decodes into a temporary FITS file."""

        input_file = tmp_path / "raw.apz"
        input_file.touch()
        result = _process_result()
        calls = {}

        def fake_unzip(filename, **options):
            """Create the decompressed filename expected by process_file."""

            calls["unzip"] = (filename, options)
            destination = Path(options["fitsdir"]) / "raw.fits"
            destination.touch()

        def fake_process(filename, **options):
            """Verify that the temporary ramp exists during processing."""

            calls["cube_file"] = Path(filename)
            assert calls["cube_file"].exists()
            return result

        monkeypatch.setattr(ap3d.apzip, "unzip", fake_unzip)
        monkeypatch.setattr(ap3d, "read_calibrations", lambda **kwargs: {})
        monkeypatch.setattr(ap3d, "process_cube", fake_process)
        monkeypatch.setattr(ap3d, "write_ap2d", lambda *args, **kwargs: None)
        ap3d.process_file(input_file, tmp_path / "out.fits")
        assert calls["unzip"][0] == str(input_file)
        assert calls["unzip"][1]["delete"] is False
        assert calls["cube_file"].suffix == ".fits"

    def test_process_file_detects_missing_apzip_output(
        self, tmp_path, monkeypatch
    ):
        """Orchestration reports when APZ decoding creates no FITS file."""

        input_file = tmp_path / "raw.apz"
        input_file.touch()
        monkeypatch.setattr(ap3d.apzip, "unzip", lambda *args, **kwargs: None)
        monkeypatch.setattr(ap3d, "read_calibrations", lambda **kwargs: {})
        with pytest.raises(RuntimeError, match="did not create expected file"):
            ap3d.process_file(input_file, tmp_path / "out.fits")


@pytest.mark.slow
class TestFullDetectorSynthetic:
    """Synthetic integration tests requiring full APOGEE detector geometry."""

    def test_process_array_fowler(self):
        """A linear science ramp reduces to the expected Fowler image."""

        signal = np.broadcast_to(
            (10 + np.arange(2048, dtype=np.float32) % 3)[None, :],
            (2048, 2048),
        )
        cube = np.stack([1000 + read * signal for read in range(4)])
        header = fits.Header(
            {"CHIP": "a", "NFRAMES": 4, "EXPTIME": 42.588}
        )
        result = ap3d.process_array(
            cube,
            header,
            detect_cosmic_rays=False,
            nfowler=1,
            use_reference=False,
        )
        np.testing.assert_allclose(result.flux, 3 * signal)
        assert result.flux.shape == (2048, 2048)
        assert result.error.shape == result.flux.shape
        assert result.mask.dtype == np.uint16
        assert result.header["BUNIT"] == "ADU"

    def test_saturation_and_bpm_masks(self):
        """Saturation and supplied BPM flags propagate into the result."""

        cube = np.zeros((4, 2048, 2048), dtype=np.float32)
        cube[:] = np.arange(4)[:, None, None] * 10
        cube[2:, 100, 200] = 65535
        bpm = np.zeros((2048, 2048), dtype=np.uint16)
        bpm[300, 400] = ap3d.PIXMASK.getval("BADPIX")
        result = ap3d.process_array(
            cube,
            fits.Header({"CHIP": "a", "NFRAMES": 4}),
            bpm=bpm,
            detect_cosmic_rays=False,
            nfowler=1,
            use_reference=False,
        )
        assert result.mask[100, 200] & ap3d.PIXMASK.getval("SATPIX")
        assert result.mask[300, 400] & ap3d.PIXMASK.getval("BADPIX")

    def test_process_array_up_the_ramp(self):
        """process_array dispatches to up-the-ramp sampling correctly."""

        signal = np.full((2048, 2048), 5.0, dtype=np.float32)
        cube = np.stack([500 + read * signal for read in range(5)])
        result = ap3d.process_array(
            cube,
            fits.Header({"CHIP": "a", "NFRAMES": 5}),
            gain=2.0,
            readnoise=10.0,
            detect_cosmic_rays=False,
            up_the_ramp=True,
            use_reference=False,
        )
        np.testing.assert_allclose(result.flux, 4 * signal)
        assert np.all(np.isfinite(result.error))
        assert "Up-the-ramp sampling" in list(result.header["HISTORY"])

    def test_reference_correction_shapes_and_masks(self):
        """Reference correction removes or retains the 512 reference columns."""

        rng = np.random.RandomState(123)
        cube = np.full((3, 2048, 2560), 10000, dtype=np.float32)
        reference = rng.normal(1000, 2, (3, 2048, 512)).astype(np.float32)
        cube[:, :, 2048:] = reference
        header = fits.Header({"SLICE000": 1, "SLICE001": 2, "SLICE002": 3})
        corrected, mask, read_mask, last_good = ap3d.reference_correct(
            cube,
            header,
            cds=False,
            vertical=False,
            horizontal=False,
            keep_reference=True,
        )
        assert corrected.shape == (3, 2048, 2560)
        assert mask.shape == (2048, 2048)
        np.testing.assert_array_equal(read_mask, [True, False, False])
        assert last_good == 2
        assert np.all(mask[:4] & ap3d.PIXMASK.getval("BADPIX"))


def _real_data_options():
    """Build process_file calibration options from AP3D_* variables."""

    mapping = {
        "detector": "AP3D_DETECTOR",
        "bpm": "AP3D_BPM",
        "dark": "AP3D_DARK",
        "flat": "AP3D_FLAT",
        "littrow": "AP3D_LITTROW",
        "persistence_mask": "AP3D_PERSISTENCE_MASK",
    }
    return {
        keyword: os.environ[variable]
        for keyword, variable in mapping.items()
        if os.environ.get(variable)
    }


@pytest.mark.realdata
def test_real_apogee_reduction(tmp_path):
    """Run the complete reduction on user-supplied APOGEE data.

    Required environment variable
    -----------------------------
    AP3D_RAWFILE
        Input ``apR-*.apz`` or decompressed 3-D FITS ramp.

    Optional calibration variables
    ------------------------------
    AP3D_DETECTOR, AP3D_BPM, AP3D_DARK, AP3D_FLAT, AP3D_LITTROW,
    AP3D_PERSISTENCE_MASK

    Optional comparison variable
    ----------------------------
    AP3D_IDL_AP2D
        Existing IDL ap2D product.  When supplied, its FLUX plane is compared
        to the Python result using robust summary diagnostics.  The comparison
        deliberately reports metrics rather than imposing a premature strict
        equality threshold.
    """

    rawfile = os.environ.get("AP3D_RAWFILE")
    if not rawfile:
        pytest.skip("Set AP3D_RAWFILE to run the real-data reduction")
    rawfile = Path(rawfile)
    if not rawfile.exists():
        pytest.fail(f"AP3D_RAWFILE does not exist: {rawfile}")
    calibrations = _real_data_options()
    for keyword, filename in calibrations.items():
        if not Path(filename).exists():
            pytest.fail(f"{keyword} calibration does not exist: {filename}")

    output = tmp_path / "python-ap2D.fits"
    result = ap3d.process_file(
        rawfile,
        output,
        overwrite=True,
        up_the_ramp=os.environ.get("AP3D_UP_THE_RAMP", "0") == "1",
        fix_cosmic_rays=True,
        detect_cosmic_rays=True,
        fix_saturation=True,
        use_reference=True,
        q3fix=False,
        verbose=True,
        debug=False,
        **calibrations,
    )
    assert output.exists()
    assert result.flux.shape == (2048, 2048)
    assert result.error.shape == result.flux.shape
    assert result.mask.shape == result.flux.shape
    assert np.all(np.isfinite(result.flux))
    assert np.all(np.isfinite(result.error))
    assert np.all(result.error >= 1)

    idl_file = os.environ.get("AP3D_IDL_AP2D")
    if idl_file:
        idl_file = Path(idl_file)
        if not idl_file.exists():
            pytest.fail(f"AP3D_IDL_AP2D does not exist: {idl_file}")
        with fits.open(idl_file, memmap=False) as hdul:
            if "FLUX" in hdul:
                idl_flux = np.asarray(hdul["FLUX"].data, dtype=np.float32)
            elif hdul[0].data is not None and hdul[0].data.ndim == 3:
                idl_flux = np.asarray(hdul[0].data[0], dtype=np.float32)
            else:
                idl_flux = np.asarray(hdul[1].data, dtype=np.float32)
        assert idl_flux.shape == result.flux.shape
        good = np.isfinite(idl_flux) & np.isfinite(result.flux)
        difference = result.flux[good] - idl_flux[good]
        median_difference = float(np.median(difference))
        robust_sigma = float(1.4826 * np.median(
            np.abs(difference - median_difference)
        ))
        relative_rms = float(
            np.sqrt(np.mean(difference**2))
            / max(np.sqrt(np.mean(idl_flux[good] ** 2)), 1.0)
        )
        print(
            "Python-IDL comparison: "
            f"median difference={median_difference:.6g}, "
            f"robust sigma={robust_sigma:.6g}, "
            f"relative RMS={relative_rms:.6g}"
        )
