"""Unit and regression tests for :mod:`apogee_drp.apred.psf`.

The suite uses synthetic profiles and detector images, so it requires no
APOGEE calibration files, database connection, or reduction-tree setup.
"""

import gc

import numpy as np
import pytest

from apogee_drp.apred import psf


def make_planar_grid(nx=3, ny=3, nprofile=7, logscale=False):
    """Create an irregular-grid-compatible profile linear in x and y."""
    xvalues = np.linspace(0.0, 2047.0, nx)
    yvalues = np.linspace(0.0, 2047.0, ny)
    xgrid, ygrid = np.meshgrid(xvalues, yvalues, indexing="ij")
    profile_dy = np.linspace(-3.0, 3.0, nprofile)

    grid = np.empty((nx, ny, nprofile), dtype=float)
    for i in range(nx):
        for j in range(ny):
            base = np.exp(-0.5 * (profile_dy / 1.1) ** 2)
            grid[i, j] = base + 1e-5 * xgrid[i, j] + 2e-5 * ygrid[i, j]

    if logscale:
        grid = np.log10(grid)

    labels = np.stack((xgrid, ygrid))
    return profile_dy, xgrid, ygrid, grid, labels


def make_two_trace_epsf(ncol=2048):
    """Create two overlapping, normalized synthetic fiber profiles."""
    epsf = []
    definitions = [(10, 4, 10, 7.1), (27, 9, 15, 11.9)]

    for fiber, lo, hi, center in definitions:
        rows = np.arange(lo, hi + 1, dtype=float)
        profile = np.exp(-0.5 * ((rows - center) / 1.05) ** 2)
        profile /= np.sum(profile)
        image = np.repeat(profile[:, None], ncol, axis=1)
        epsf.append(
            {
                "fiber": fiber,
                "lo": lo,
                "hi": hi,
                "img": image,
                "ycen": np.full(ncol, center),
            }
        )

    return epsf


def render_small_frame(epsf, spectra, nrow=20):
    """Render an EPSF list into a small detector image."""
    ncol = spectra.shape[0]
    image = np.zeros((nrow, ncol), dtype=float)

    for element in epsf:
        fiber = element["fiber"]
        image[element["lo"]:element["hi"] + 1] += (
            element["img"] * spectra[:, fiber][None, :]
        )

    return image


class TestElementaryFunctions:
    def test_leaky_relu(self):
        values = np.array([-3.0, -0.0, 0.0, 2.5])
        expected = np.array([-0.03, 0.0, 0.0, 2.5])
        np.testing.assert_allclose(psf.leaky_relu(values), expected)

    @pytest.mark.parametrize(
        "pars,expected",
        [
            (np.array([2.0, 3.0, -4.0]), np.array([2.0, 3.0, -4.0])),
            (np.array([2.0, 3.0, 5.0, -4.0]), np.array([2.0, 3.0, 5.0, -4.0])),
        ],
    )
    def test_func_poly2d_numba(self, pars, expected):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([4.0, 5.0, 6.0])

        if len(pars) == 3:
            answer = expected[0] + expected[1] * x + expected[2] * y
        else:
            answer = (
                expected[0]
                + expected[1] * x
                + expected[2] * x * y
                + expected[3] * y
            )

        np.testing.assert_allclose(
            psf.func_poly2d_numba(x, y, pars), answer
        )

    def test_func_poly2d_numba_rejects_unsupported_parameter_count(self):
        x = np.array([1.0])
        y = np.array([2.0])
        with pytest.raises(Exception, match="Only 0, 3, and 4"):
            psf.func_poly2d_numba(x, y, np.ones(2))

    def test_python_and_numba_polynomial_functions_agree(self):
        x = np.array([0.0, 10.0, 100.0])
        y = np.array([20.0, 30.0, 40.0])
        pars = np.array([0.2, -1e-3, 2e-6, 3e-4])
        expected = psf.func_poly2d([x, y], *pars)
        actual = psf.func_poly2d_numba(x, y, pars)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


class TestGridInterpolation:
    @pytest.mark.parametrize(
        "labels,index",
        [
            ((0.0, 0.0), (0, 0)),
            ((0.0, 2047.0), (0, -1)),
            ((2047.0, 0.0), (-1, 0)),
            ((2047.0, 2047.0), (-1, -1)),
        ],
    )
    def test_gridinterp_corners(self, labels, index):
        _, xgrid, ygrid, grid, _ = make_planar_grid()
        actual = psf.gridinterp(np.array(labels), xgrid, ygrid, grid)
        np.testing.assert_allclose(actual, grid[index[0], index[1]])

    @pytest.mark.parametrize(
        "labels",
        [
            (0.0, 700.0),
            (2047.0, 1300.0),
            (600.0, 0.0),
            (1500.0, 2047.0),
            (700.0, 1300.0),
        ],
    )
    def test_gridinterp_edges_and_interior_are_exact_for_planar_grid(self, labels):
        profile_dy, xgrid, ygrid, grid, _ = make_planar_grid()
        base = np.exp(-0.5 * (profile_dy / 1.1) ** 2)
        expected = base + 1e-5 * labels[0] + 2e-5 * labels[1]
        actual = psf.gridinterp(np.array(labels), xgrid, ygrid, grid)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("labels", [(-1.0, 10.0), (10.0, -1.0), (2048.0, 10.0), (10.0, 2048.0)])
    def test_gridinterp_rejects_out_of_detector_coordinates(self, labels):
        _, xgrid, ygrid, grid, _ = make_planar_grid()
        with pytest.raises(ValueError, match="between 0 and 2047"):
            psf.gridinterp(np.array(labels), xgrid, ygrid, grid)


class TestFiberProfileConstruction:
    def test_build_fiber_epsf_shape_and_normalization(self):
        profile_dy, xgrid, ygrid, grid, _ = make_planar_grid()
        trace_y = np.array([100.2, 100.4, 100.6, 100.8])
        detector_y = np.arange(96, 106, dtype=float)
        result = psf.build_fiber_epsf(
            trace_y, detector_y, profile_dy, xgrid, ygrid, grid, False
        )

        assert result.shape == (detector_y.size, trace_y.size)
        assert np.all(result >= 0.0)
        np.testing.assert_allclose(np.sum(result, axis=0), 1.0, atol=1e-14)

    def test_log_and_linear_profile_grids_agree(self):
        profile_dy, xgrid, ygrid, linear_grid, _ = make_planar_grid()
        # Use a spatially invariant profile here. Interpolation in log space
        # is not generally identical to interpolation in linear space when
        # the grid itself varies spatially.
        linear_grid[:] = linear_grid[0, 0]
        log_grid = np.log10(linear_grid)
        trace_y = np.array([500.0, 500.0, 500.0])
        detector_y = 500.0 + profile_dy
        
        linear = psf.build_fiber_epsf(
            trace_y, detector_y, profile_dy, xgrid, ygrid,
            linear_grid, False
        )
        logarithmic = psf.build_fiber_epsf(
            trace_y, detector_y, profile_dy, xgrid, ygrid,
            log_grid, True
        )
        np.testing.assert_allclose(logarithmic, linear, rtol=2e-3, atol=2e-4)

    def test_nonpositive_profile_sum_is_rejected(self):
        profile_dy, xgrid, ygrid, grid, _ = make_planar_grid()
        grid[:] = 0.0
        with pytest.raises(ValueError, match="non-positive sum"):
            psf.build_fiber_epsf(
                np.array([100.0]), np.arange(97.0, 104.0), profile_dy,
                xgrid, ygrid, grid, False
            )

    def test_build_epsf_grid_subset_offsets_and_detector_clipping(self):
        profile_dy, xgrid, ygrid, grid, _ = make_planar_grid()
        ncol = 8
        trace_y = np.vstack(
            (
                np.linspace(5.0, 5.5, ncol),
                np.linspace(100.0, 101.0, ncol),
                np.linspace(2041.0, 2042.0, ncol),
            )
        )
        fibers = np.array([0, 2], dtype=np.int64)
        offset = np.array([0.25, 0.0, 0.0, 0.0])

        cube, centers, row_start, row_stop = psf.build_epsf_grid(
            trace_y, fibers, offset, profile_dy, xgrid, ygrid, grid, False
        )

        assert cube.shape == (2, 100, ncol)
        np.testing.assert_allclose(centers, trace_y[fibers] + 0.25)
        assert row_start[0] == 0
        assert row_stop[1] == 2047

        for i in range(2):
            height = row_stop[i] - row_start[i] + 1
            np.testing.assert_allclose(
                np.sum(cube[i, :height], axis=0), 1.0, atol=1e-14
            )

    def test_build_epsf_grid_rejects_profiles_taller_than_storage(self):
        profile_dy, xgrid, ygrid, grid, _ = make_planar_grid()
        trace_y = np.array([np.linspace(100.0, 300.0, 8)])
        with pytest.raises(ValueError, match="more than 100"):
            psf.build_epsf_grid(
                trace_y, np.array([0]), np.zeros(4), profile_dy,
                xgrid, ygrid, grid, False
            )


class TestPSFProfile:
    @pytest.fixture
    def profile(self):
        x = np.linspace(-5.0, 5.0, 101)
        y = 2.0 + 0.3 * x + 0.05 * x**2
        return psf.PSFProfile(x, y)

    def test_quadratic_interpolation_is_exact(self, profile):
        requested = np.linspace(-4.95, 4.95, 73)
        expected = 2.0 + 0.3 * requested + 0.05 * requested**2
        np.testing.assert_allclose(profile(requested), expected, atol=1e-11)

    def test_values_outside_profile_are_zero(self, profile):
        np.testing.assert_array_equal(profile(np.array([-6.0, 6.0])), [0.0, 0.0])

    def test_copy_is_independent(self, profile):
        copied = profile.copy()
        copied.y[0] += 100.0
        assert copied.y[0] != profile.y[0]

    @pytest.mark.parametrize("operation,value", [("add", 3.0), ("sub", 3.0), ("mul", 2.0), ("div", 2.0)])
    def test_scalar_arithmetic(self, profile, operation, value):
        requested = np.linspace(-4.0, 4.0, 21)
        original = profile(requested)
        if operation == "add":
            result = profile + value
            expected = original + value
        elif operation == "sub":
            result = profile - value
            expected = original - value
        elif operation == "mul":
            result = profile * value
            expected = original * value
        else:
            result = profile / value
            expected = original / value
        np.testing.assert_allclose(result(requested), expected, atol=1e-11)

    def test_profile_addition(self, profile):
        result = profile + profile
        requested = np.linspace(-4.0, 4.0, 21)
        np.testing.assert_allclose(result(requested), 2.0 * profile(requested))

    def test_mismatched_profiles_are_rejected(self, profile):
        other = psf.PSFProfile(np.linspace(-4.0, 4.0, 101), np.ones(101))
        with pytest.raises(Exception, match="X arrays must be the same"):
            _ = profile + other


class TestPSFGridObject:
    @pytest.fixture
    def grid_psf(self):
        profile_dy, _, _, grid, labels = make_planar_grid()
        return psf.PSF((grid, labels, profile_dy), kind="grid", log=False)

    def test_grid_model_matches_low_level_interpolator(self, grid_psf):
        labels = np.array([700.0, 1300.0])
        expected = psf.gridinterp(
            labels, grid_psf._xgrid, grid_psf._ygrid, grid_psf._grid
        )
        np.testing.assert_allclose(grid_psf.model(labels), expected)

    def test_call_resamples_to_detector_rows(self, grid_psf):
        labels = np.array([700.0, 1000.25])
        detector_y = np.arange(996.0, 1006.0)
        output = grid_psf(labels, y=detector_y, ycen=labels[1])
        expected = np.interp(
            detector_y - labels[1], grid_psf.y, grid_psf.gridinterp(labels),
            left=grid_psf.gridinterp(labels)[0],
            right=grid_psf.gridinterp(labels)[-1],
        )
        np.testing.assert_allclose(output, expected)

    def test_buildepsf_supports_fiber_subset(self, grid_psf):
        ncol = 8
        traces = np.vstack(
            (
                np.linspace(100.0, 100.5, ncol),
                np.linspace(500.0, 500.5, ncol),
                np.linspace(900.0, 900.5, ncol),
            )
        )
        requested = np.array([0, 2], dtype=np.int64)
        output = grid_psf.buildepsf(
            traces,
            fibers=requested,
            offcoef=np.zeros(4),
        )

        assert len(output) == 2
        np.testing.assert_array_equal(
            [element["fiber"] for element in output], requested
        )
        for element in output:
            np.testing.assert_allclose(
                np.sum(element["img"], axis=0), 1.0, atol=1e-14
            )

    def test_repr_contains_kind_and_profile_size(self, grid_psf):
        representation = repr(grid_psf)
        assert "grid" in representation
        assert "Npix=7" in representation

    def test_invalid_kind_is_rejected(self):
        with pytest.raises(ValueError, match="Only .*ann.*grid"):
            psf.PSF(None, kind="invalid")

    def test_out_of_detector_call_is_rejected(self, grid_psf):
        with pytest.raises(ValueError, match="between 0 and 2047"):
            grid_psf(np.array([-1.0, 100.0]))


class TestExtractionHelpers:
    def test_extract_pmul_matches_direct_overlap(self):
        ncol = 2048
        first = {
            "lo": 4,
            "hi": 8,
            "img": np.repeat(np.arange(1.0, 6.0)[:, None], ncol, axis=1),
        }
        second = {
            "lo": 6,
            "hi": 10,
            "img": np.repeat(np.arange(2.0, 7.0)[:, None], ncol, axis=1),
        }
        image = first["img"].T
        actual = psf.extract_pmul(first["lo"], first["hi"], image, second)
        expected_value = np.sum(np.array([3.0, 4.0, 5.0]) * np.array([2.0, 3.0, 4.0]))
        np.testing.assert_allclose(actual, expected_value)

    def test_extract_pmul_no_overlap_returns_zero(self):
        ncol = 2048
        image = np.ones((ncol, 3))
        second = {"lo": 10, "hi": 12, "img": np.ones((3, ncol))}
        actual = psf.extract_pmul(0, 2, image, second)
        np.testing.assert_array_equal(actual, np.zeros(ncol))

    def test_solvefibers_back_substitution(self):
        # Upper-triangular system after forward elimination.
        b = np.array([2.0, 3.0, 4.0])
        c = np.array([0.5, -0.25, 0.0])
        truth = np.array([1.5, -2.0, 3.0])
        v = np.array([
            b[0] * truth[0] + c[0] * truth[1],
            b[1] * truth[1] + c[1] * truth[2],
            b[2] * truth[2],
        ])
        x = np.zeros(3)
        x[-1] = truth[-1]
        xvar = np.zeros(3)
        xvar[-1] = 0.25
        vvar = np.array([1.0, 1.0, 4.0])
        result, result_var = psf.solvefibers(x, xvar, 3, v, b, c, vvar)
        np.testing.assert_allclose(result, truth)
        assert np.all(result_var >= 0.0)

    def test_epsfmodel_clips_nonpositive_spectra_and_honors_filters(self):
        epsf = make_two_trace_epsf()
        spectra = np.zeros((2048, 300))
        spectra[:, 10] = 5.0
        spectra[:, 27] = -3.0

        all_model = psf.epsfmodel(epsf, spectra)
        # These values are indices into the EPSF list, not fiber IDs.
        only_second = psf.epsfmodel(
            epsf, spectra, subonly=np.array([1])
        )
        skip_first = psf.epsfmodel(
            epsf, spectra, skip=np.array([0])
        )
        assert np.max(all_model) > 0.0
        np.testing.assert_array_equal(only_second, 0.0)
        np.testing.assert_array_equal(skip_first, 0.0)


class TestSolveAllColumns:
    def make_system(self, nsystem=3, ncol=12):
        tridiag = np.zeros((3, nsystem, ncol), dtype=float)
        beta = np.zeros((nsystem, ncol), dtype=float)
        betavar = np.ones((nsystem, ncol), dtype=float)
        psftot = np.ones((nsystem, ncol), dtype=float)
        warn = np.zeros((nsystem, ncol), dtype=np.int64)
        bad = np.zeros((nsystem, ncol), dtype=np.int64)
        return tridiag, beta, betavar, psftot, warn, bad

    def test_matches_dense_tridiagonal_solution(self):
        tridiag, beta, betavar, psftot, warn, bad = self.make_system()
        fibers = np.array([5, 2, 11], dtype=np.int64)
        truth = np.array([3.0, -1.5, 4.0])
        matrix = np.array([[2.5, 0.2, 0.0], [0.2, 3.0, -0.3], [0.0, -0.3, 2.0]])

        tridiag[1] = matrix.diagonal()[:, None]
        tridiag[0, 1:] = np.array([matrix[1, 0], matrix[2, 1]])[:, None]
        tridiag[2, :-1] = np.array([matrix[0, 1], matrix[1, 2]])[:, None]
        beta[:] = (matrix @ truth)[:, None]

        spec, err, mask, back = psf.solve_all_columns(
            tridiag, beta, betavar, psftot, fibers, warn, bad, nout=20
        )

        for column in range(4, 8):
            np.testing.assert_allclose(spec[column, fibers], truth, atol=1e-12)
            np.testing.assert_array_equal(mask[column, fibers], 0)
        np.testing.assert_array_equal(back, 0.0)

    def test_diagonal_system_uncertainties(self):
        tridiag, beta, betavar, psftot, warn, bad = self.make_system(nsystem=2)
        fibers = np.array([1, 7], dtype=np.int64)
        diagonal = np.array([2.0, 4.0])
        tridiag[1] = diagonal[:, None]
        beta[:] = np.array([6.0, 20.0])[:, None]
        betavar[:] = np.array([9.0, 16.0])[:, None]

        spec, err, _, _ = psf.solve_all_columns(
            tridiag, beta, betavar, psftot, fibers, warn, bad, nout=10
        )
        np.testing.assert_allclose(spec[4, fibers], [3.0, 5.0])
        np.testing.assert_allclose(err[4, fibers], [1.5, 1.0])

    def test_rejected_middle_trace_is_disconnected_and_masked(self):
        tridiag, beta, betavar, psftot, warn, bad = self.make_system()
        fibers = np.array([0, 1, 2], dtype=np.int64)
        tridiag[1] = 2.0
        tridiag[0, 1:] = 0.7
        tridiag[2, :-1] = 0.7
        beta[0] = 6.0
        beta[1] = 100.0
        beta[2] = 10.0
        psftot[1, 6] = 0.2
        bad[1, 6] = 4
        warn[0, 6] = 32

        spec, _, mask, _ = psf.solve_all_columns(
            tridiag, beta, betavar, psftot, fibers, warn, bad, nout=3
        )

        assert spec[6, 0] == pytest.approx(3.0)
        assert spec[6, 2] == pytest.approx(5.0)
        assert spec[6, 1] == 0.0
        assert mask[6, 0] == 32
        assert mask[6, 1] == (16384 | 4)

    def test_no_good_components_sets_bad_errors_and_masks(self):
        tridiag, beta, betavar, psftot, warn, bad = self.make_system(nsystem=2)
        fibers = np.array([3, 8], dtype=np.int64)
        tridiag[1] = 1.0
        psftot[:, 5] = 0.0
        bad[:, 5] = np.array([1, 4])

        spec, err, mask, _ = psf.solve_all_columns(
            tridiag, beta, betavar, psftot, fibers, warn, bad, nout=10
        )
        np.testing.assert_array_equal(spec[5], 0.0)
        np.testing.assert_array_equal(err[5], psf.BADERR)
        assert mask[5, 3] == (16384 | 1)
        assert mask[5, 8] == (16384 | 4)

    def test_background_component_is_returned_separately(self):
        tridiag, beta, betavar, psftot, warn, bad = self.make_system(nsystem=3)
        fibers = np.array([4, 9], dtype=np.int64)
        tridiag[1] = np.array([2.0, 4.0, 5.0])[:, None]
        beta[:] = np.array([6.0, 20.0, 35.0])[:, None]

        spec, _, _, back = psf.solve_all_columns(
            tridiag, beta, betavar, psftot, fibers, warn, bad,
            doback=True, nout=12
        )
        np.testing.assert_allclose(spec[4, fibers], [3.0, 5.0])
        assert back[4] == pytest.approx(7.0)

    def test_four_pixel_detector_edges_remain_unprocessed(self):
        tridiag, beta, betavar, psftot, warn, bad = self.make_system(nsystem=1)
        fibers = np.array([2], dtype=np.int64)
        tridiag[1] = 1.0
        beta[0] = 12.0

        spec, err, mask, _ = psf.solve_all_columns(
            tridiag, beta, betavar, psftot, fibers, warn, bad, nout=5
        )
        edge = np.array([0, 1, 2, 3, 8, 9, 10, 11])
        np.testing.assert_array_equal(spec[edge], 0.0)
        np.testing.assert_array_equal(err[edge], 999999.09)
        np.testing.assert_array_equal(mask[edge], 1)


class TestExtractionRegression:
    def test_extract2_matches_original_extract_on_synthetic_frame(self):
        ncol = 2048
        epsf = make_two_trace_epsf(ncol=ncol)
        truth = np.zeros((ncol, 300), dtype=float)
        phase = np.linspace(0.0, 2.0 * np.pi, ncol)
        truth[:, 10] = 800.0 + 50.0 * np.sin(phase)
        truth[:, 27] = 350.0 + 30.0 * np.cos(phase)
        image = render_small_frame(epsf, truth)
        frame = {
            "flux": image,
            "err": np.ones_like(image),
            "mask": np.zeros_like(image, dtype=np.int64),
            "header": {"TEST": True},
        }

        oldout, oldback, oldmodel = psf.extract(frame, epsf)
        newout, newback, newmodel = psf.extract2(frame, epsf)
        columns = slice(4, 2044)
        fibers = np.array([10, 27])

        np.testing.assert_allclose(
            newout["flux"][columns, fibers],
            oldout["flux"][columns, fibers],
            rtol=1e-11,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            newout["err"][columns, fibers],
            oldout["err"][columns, fibers],
            rtol=1e-11,
            atol=1e-10,
        )
        np.testing.assert_array_equal(newout["mask"], oldout["mask"])
        np.testing.assert_allclose(newback, oldback)
        np.testing.assert_allclose(newmodel, oldmodel, rtol=1e-12, atol=1e-12)

        # The noiseless synthetic extraction should also recover the truth.
        np.testing.assert_allclose(
            newout["flux"][columns, fibers],
            truth[columns, fibers],
            rtol=1e-11,
            atol=1e-9,
        )

        del oldmodel, newmodel, oldout, newout
        gc.collect()
