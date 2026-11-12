Native and apStar polynomial LSFs
=================================

The native calibration fits one shared **pixel-based Gauss--Hermite LSF per
fiber** across all three detectors. The apStar LSF is then calculated by
propagating that solution through the visit-to-apStar resampling. Calibration
measurements are fitted only in the native stage.

Native pixel coordinates
-------------------------

The native model follows the standard APOGEE GH convention:

* Profile offsets, core sigma, wing sigma and integration bins are in native
  detector pixels. Native pixel integration has width one.
* Polynomial variation uses a common coordinate ``X = pixel + chip_offset``.
  The measured chip offsets are taken from the first wavelength-solution
  coefficient, matching the historical ``APLSF`` global fit (``fitmethod=3``).
* Each chip exports the same polynomial coefficients with its own ``Xoffset``.
  This is an exact algebraic conversion, without a wavelength projection or
  another fit. The standard per-chip GH evaluator can use these vectors.

Wavelength is used to identify calibration lines and doublet separations and,
subsequently, to map between spectra. It is not the argument of the native GH
profile or its coefficient polynomials. Internally the polynomials are
conditioned with a Chebyshev basis; exported vectors use ordinary powers, as
expected by the existing APOGEE implementation.

Fitting and writing native calibration files
---------------------------------------------

.. code-block:: python

    from apogee_drp.utils.native_lsf import NativeLSFCalibration

    files = [f"/calibration/ap1D-{c}-40030031.fits" for c in "abc"]
    calibration = NativeLSFCalibration.fit(
        files,
        line_catalog="/path/to/apogee_drp/data/skylines/airglow.txt",
        degree=3, horder=5,
    )
    calibration.write(
        files,
        [f"/new-calibration/apLSF-{c}-40030031.fits" for c in "abc"],
        global_output="/new-calibration/apLSF-40030031.fits",
    )
    pixel_model = calibration.model(22)
    red_chip_parameters = pixel_model.parameters(chip=0)

Inputs are wavelength-calibrated native ap1D files in a/b/c order, with the
wavelength coefficient arrays in HDU5. An explicit chip-offset array can be
supplied to the lower-level ``fit_native_pixel`` interface for alternate formats.
An old IDL ``.sav`` fit is unnecessary; the airglow catalog supplies wavelengths
and doublet separations. ``saved_fit=...`` retains the earlier line-selection
comparison option.

All 300 fibers are attempted by default. ``fibers=[22, 23, 150]`` selects
physical fiber IDs, with array row ``300 - fiber``. Fits are independent between
fibers. ``on_error="record"`` records failures; ``on_error="raise"`` aborts on
the first failed fiber. By default failed native-pixel fits are estimated using
a local linear fit through the five nearest measured fibers. Each donor's
coefficient polynomial is first put on a common pixel basis. Filled fibers are
never used as donors. At an edge the method uses the nearest measured fiber
instead of extrapolating. ``fill_missing=False`` leaves failures explicitly
unavailable; unrequested fibers remain NaN in a subset calibration.

The measured line fits include pixel integration, catalog doublets, free line
centers, amplitudes and linear backgrounds. Hard bad pixels and nearby blends
are masked. Every fourth accepted line is held out, with shared wing pixels
removed from training to avoid leakage. Held-out LSF coefficients remain fixed
while line-specific nuisance values are fitted. Raw errors are not rescaled to
force chi-square to one. Physical width/fraction constraints cover the complete
detector range, including regions beyond the outermost measured lines.

The default native H5 cubic model has 24 LSF coefficients: four each for core
sigma, H3, H4, H5, wing fraction and wing sigma. H1 and H2 are fixed at zero;
line centers are fitted independently. Line-specific nuisance parameters are
not included in that LSF coefficient count. A different degree can be chosen
for selected fibers with ``degree_overrides={48: 4, 99: 4}``; other fibers keep
the specified common degree. The builder exposes this as
``native_degree_overrides``. The format supports different polynomial degrees
between fibers without changing the pixel-coordinate convention.

``examples/build_native_lsf.py`` provides the standalone command-line interface.
The usual ``apred.cal.lsf.build_lsf`` entry point retains raw-exposure preparation
and same-dither combination, then uses the shared native-pixel fitter by default.
``global_fit=False`` selects the previous per-chip Gaussian or ``full=True`` GH
path. Different calibration dither positions require separate preparation.

Filling missing fibers
-----------------------

A calibration that was fitted previously can be completed explicitly:

.. code-block:: python

    calibration.fill_missing(method="local_linear", neighbors=5, files=files)
    print(calibration.filled_fibers)

``method="linear"`` instead uses the nearest bracketing pair;
``method="nearest"`` uses one measured neighbor. ``method="polynomial"``
allows a specified local ``degree`` and neighbor count. The default local
linear method outperformed direct copying, two-neighbor interpolation and the
tested higher-order local fits in leave-one-fiber-out reconstruction tests on
the supplied calibration. These compare to independently fitted LSF profiles,
not to a known true LSF; estimated fibers do not have the precision guarantee
of measured ones.

The usual builder exposes ``native_fill_missing``, ``native_fill_method`` and
``native_fill_neighbors``. The standalone example accepts ``--no-fill`` to
retain unavailable fibers. The retained wavelength-based mode does not fill
failures by default.

Filled models have ``filled=True``, their source fibers/weights, method and
original failure in their diagnostics. Valid target chip geometry is used when
available; otherwise geometry is estimated from the donors and flagged. This
fills the LSF only and does not repair an invalid wavelength calibration.

Calibration file contents
--------------------------

The three standard ``apLSF-[abc]`` files retain their leading HDUs:

* Primary: the standard packed pixel-coordinate parameter array.
* First extension: the full pixel-integrated ``LSF ARRAY`` cube, indexed as
  ``[LSF offset, fiber row, native pixel]`` in Python.

Additional extensions store ``GLOBAL_LSF`` and ``FIBER_STATUS``. The latter
includes ``FILLED``, ``ORIGINAL_STATUS`` and ``SOURCES`` alongside ``STATUS``.
Filled entries have statuses such as ``FILLED_LOCAL_LINEAR``. The global model
retains the full provenance when embedded in an apVisit.
For native-pixel models, ``GLOBAL_LSF`` version 2 explicitly declares
``COORDSYS=NATIVE_PIXEL`` and stores the shared polynomial, chip offsets and
native array size. ``PIXEXACT=True`` marks exact standard-pixel exports.
``PROJECTION_QA`` contains zero projection errors for this algebraic export.

The optional combined file stores the exact solutions and provenance. The
normal builder uses the registered fourth LSF product, historically named
``*-diagnostics.fits``, for this information. Existing products are protected
unless overwrite is requested; the three chip files are staged before writing.
The supplied calibration now contains 290 measured and ten estimated fiber
solutions, with finite coefficients and profile arrays for all 300 fibers.

The Python visit writer appends the appropriate fiber's global model to each
apVisit without changing historical HDU indices. The existing dither combiner
converts the standard pixel coefficients to half-pixel coordinates. Core and
wing widths, polynomial powers, Xoffset and bin size transform algebraically.

Converting to the apStar scale
-------------------------------

``rv.visitcomb`` reads the embedded native model, maps wavelengths using the
Doppler shift actually applied to that visit, evaluates the standard native
pixel GH profiles and passes them through the actual chip-specific damped-sinc
resampler. It then fits a compact polynomial representation of the resulting
apStar profiles. ``VREL`` is the visit-to-rest shift; barycentric ``VRAD`` cannot
be substituted for it. Barycentric-only combination uses its actual applied
shift as well.

One LSF table entry is saved per apStar flux row:

* Row 0: the combination with the actual per-pixel weights.
* Row 1, for multiple visits: the combination with its smoothed-error weights.
* Rows 2 onward: the individual resampled visits. A single-visit apStar has
  only row 0.

Zero-error or otherwise invalid samples contribute zero combination weight.
The spectrum and LSF use the same weighting arrays. Classical visits without
``GLOBAL_LSF`` are still supported through their stored native pixel vectors.
An explicit ``native_calibration`` model, or one model per visit, can also be
passed to ``derive_apstar_lsf``.

HDU8 ``LSF TABLE`` contains ``ROW``, ``KIND``, ``PARAMETERS``, ``NPARS``,
``DOMAINS``, ``MAXERR`` and ``STATUS``. ``ROW`` is the zero-based flux-row index.
``DOMAINS`` excludes chip gaps and unsupported edges. ``MAXERR`` measures
held-out profile residuals relative to the reference LSF peak, not the stellar
continuum. ``WARN`` flags errors above a provisional 1% limit. ``NO_RV``,
``NO_LSF`` and ``NO_DATA`` identify unavailable entries without inventing a
replacement LSF or discarding the spectrum. ``NATIVE_FILLED`` separately flags
rows that use an estimated native LSF; combined rows are flagged if any
contributing visit uses one. ``LSFFILL`` is also recorded in the apVisit and
apStar primary headers. ``MAXERR`` measures only the resampled approximation;
it does not include the additional uncertainty of filling a missing fiber.

The default approximation keeps H5 and uses mixed polynomial degrees
``(3, 0, 4, 1, 2, 2, 4)`` for sigma, H1, H3, H4, H5, wing fraction and wing sigma.
That gives 23 fitted coefficients. H2 remains zero. This choice reached 0.93%
maximum profile error on a fresh 1,053-profile test set for the supplied fiber.
The previous 18-coefficient choice is retained as ``H5_MIXED_DEGREES`` for
comparisons; it did not retain 1% accuracy for this pixel-native model.

For strict rejection use ``lsf_options={"max_error": 0.01}`` in ``visitcomb``.
Other polynomial orders and GH orders can be supplied through ``lsf_options``;
use ``polynomial_degrees=None`` for a uniform degree. ``fit_lsf=False`` disables
conversion. Saved profiles already include native integration and resampling;
do not apply a second pixel response. Negative sinc lobes are retained.

.. code-block:: python

    from apogee_drp.utils.resampled_lsf import evaluate_lsf_table

    profiles = evaluate_lsf_table(
        apstar.lsftab, row=0, centers=[4000.3], offsets=range(-16, 17),
    )

Retained wavelength-based alternative and validation
-----------------------------------------------------

``NativeLSFCalibration.fit(..., coordinate="log_wavelength")`` retains the
earlier shared wavelength-based fit. Existing version-1 ``GLOBAL_LSF`` files
and ``NativePolynomialLSF`` NPZ files remain readable. For that model only,
standard per-chip pixel vectors are approximations with measured projection
errors and ``PIXEXACT=False``. Coordinate formats are explicitly identified;
a wavelength solution is never interpreted as a pixel solution.

On the supplied calibration exposure, both coordinate models fitted the same
290 fibers and flagged ten. The median pixel/wavelength held-out chi-square
ratio was 0.994; five fibers worsened by more than 10%. Similar chi-square does
not establish absolute calibration accuracy. Fits with large residuals still
need inspection. The new native export matches standard GH evaluation to
floating-point precision; the apStar polynomial is a separate approximation.

Both versions remain experimental. Other exposures, fibers, full reduction
stages and downstream science fits need validation. Profile-peak errors alone
do not certify radial-velocity or abundance accuracy. Nonlinear continuum
estimation and bad-pixel replacement are outside this linear LSF model.

Checking filled fibers against calibration peaks
------------------------------------------------

Filled fibers with usable calibration flux now receive a sigma-only refit
against their own isolated peaks. The existing polynomial in native pixel
position is retained (cubic by default); Hermite coefficients and wing
parameters remain at their neighboring-fiber estimates. Line centers,
amplitudes and backgrounds are fitted separately. This permits a useful width
constraint even when a detector has too few lines for a full LSF fit.

Only training peaks determine sigma. Held-out peaks guard against worsening
residuals, and full-detector width limits guard against unsupported edge swings.
The default allowance is the larger of 2% of the original validation chi-square
and one approximate noise fluctuation, ``sqrt(2*dof)``. This is an adoption
heuristic, not a formal significance test. A rejected correction or insufficient
data leaves the estimated solution unchanged. A successful width check does
not establish that the estimated Hermite terms fit the data adequately.

This check runs automatically during native calibration construction.
``native_check_filled_peaks=False`` disables it in ``build_lsf``;
``native_sigma_options`` supplies refinement settings. Existing calibrations
can be checked using ``calibration.check_filled_peaks(ap1d_files)``. Repeating
the call skips already checked fibers unless ``force=True`` is supplied.

The global model stores before/after validation and sigma curves in its
diagnostics. ``FIBER_STATUS`` adds ``PEAK_CHECK`` and ``SIGMA_REFINED``;
``NO_USABLE_DATA`` explicitly identifies fibers without usable flux.
``SIGREFIN`` in visit/apStar headers and ``NATIVE_SIGMA_REFINED`` in the
apStar LSF table indicate use of a data-constrained sigma. ``LSFFILL`` and
``NATIVE_FILLED`` remain set because the other shape parameters are estimated.

To save diagnostic PDFs while checking filled fibers:

.. code-block:: python

    calibration.check_filled_peaks(
        ap1d_files, diagnostic_directory="lsf-diagnostics",
    )

For a saved calibration that has already been checked, use
``calibration.plot_filled_peaks(ap1d_files, "lsf-diagnostics")``. This writes one
PDF per filled fiber: sigma versus native detector pixel, training/held-out
peak locations, and observed peak overlays with the original and adopted LSFs.
Rejected candidate sigma curves are labeled; no-data fibers get a status page.
Plotting requires matplotlib and does not change the calibration. The builder
example exposes the same option as ``--diagnostic-directory``; ``build_lsf``
accepts it through ``native_sigma_options={"diagnostic_directory": "lsf-diagnostics"}``.

Run-wide diagnostic overview
----------------------------

The native-pixel calibration builder and ``build_native_lsf.py`` automatically
save ``apLSF-<id>-diagnostics.pdf`` alongside the calibration products. This
covers every attempted fiber, with fit status, held-out observed-peak residuals,
validation coverage and maps of sigma and all fitted LSF shape fields across
fiber and native detector position. Inspect discontinuities, isolated outliers,
large residuals and gaps. Blank cells mean unavailable models or validation;
they do not indicate a perfect fit. Filled fibers use their own peak-check
statistics rather than donor residuals. Residuals alone are not a science-quality
certification.

To regenerate the overview from a saved native-pixel calibration:

.. code-block:: python

    calibration = NativeLSFCalibration.read("apLSF-40030031.fits")
    calibration.plot_diagnostics("apLSF-40030031-diagnostics.pdf")

The overview requires matplotlib and no raw spectra. The separate detailed
filled-fiber PDFs remain optional.
