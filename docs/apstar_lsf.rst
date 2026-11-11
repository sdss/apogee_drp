Polynomial LSF calibration and apStar conversion
================================================

The LSF parameters vary as polynomials across the full wavelength range. One
shared solution is fitted directly to measured line profiles across all three
native detectors; it can then be propagated through the actual visit-to-apStar
resampling and represented by one packed polynomial coefficient vector.

Fitting the original calibration data
-------------------------------------

.. code-block:: python

    from apogee_drp.utils.calibration_lsf import (
        load_calibration_windows, fit_native_polynomial,
        evaluate_held_out, NativePolynomialLSF,
    )

    directory = "/path/to/calibration"
    files = [f"{directory}/ap1D-{c}-40030031.fits" for c in "abc"]
    windows = load_calibration_windows(
        files, f"{directory}/apLSF-40030031.sav", fiber=22,
    )
    native = fit_native_polynomial(windows, degree=3, horder=5)
    metrics, predictions, held_out = evaluate_held_out(native, windows)
    native.save("native-fiber22.npz")

The ap1D files supply paired measured flux, uncertainty, mask and wavelength
arrays. The IDL save file supplies line identities, approximate positions and
OH-doublet separations. Saved LSF coefficients and line-fit amplitudes are not
treated as observations or as authoritative values. The loader selects lines
using their measured signal-to-noise, masks nearby line cores and hard bad
pixels, and excludes shared validation pixels from the training set.

The common coordinate is ``(log10(wavelength)-4.179)/6e-6`` in the observed
calibration frame. Gauss--Hermite core and Gaussian wing parameters are smooth
polynomials in this coordinate. Integration uses the actual native pixel
edges, so the varying dispersion and reversed native axes are handled directly.
Native H1 and H2 are zero to define the centroid and width; every line has its
own fitted center, amplitude and linear background. Known OH doublets use the
equal-component convention of the original calibration code.

Amplitudes and backgrounds are solved linearly inside the nonlinear fit.
Analytic derivatives include these nuisance solutions, reducing the cost of
fitting the shared LSF. Raw uncertainty arrays are not rescaled to force a
reduced chi-square of one. Held-out line centers/amplitudes/backgrounds are
fitted independently while keeping the LSF coefficients fixed.

This is an experimental comparison on selected line windows, not a replacement
for the full sky-line-template calibration pipeline. Inspect held-out residuals,
blends, uncertainty quality and extrapolation before using a new calibration.

Converting to the apStar scale
------------------------------

.. code-block:: python

    from apogee_drp.utils.lsf import derive_apstar_lsf
    from apogee_drp.utils.apstar_lsf import ApStarLSF

    result = derive_apstar_lsf(
        "apStar-example.fits", ["apVisit-example.fits"],
        native_calibration=native, degree=3, horder=8,
    )
    result.save("apstar-lsf.fits")
    lsf = ApStarLSF.read("apstar-lsf.fits")
    packed_coefficients = lsf.parameters    # One legacy-format polynomial vector
    kernels = lsf([1500.25, 4500.65, 7000.0])

The native model describes the continuous instrumental response before pixel
integration and resampling. It is not itself the effective apStar LSF. The
converter evaluates it at the visit-frame wavelength, integrates over native
pixels, and applies the exact chip-specific damped-sinc interpolation used by
``rv.visitcomb``. It then fits all output reference profiles simultaneously
with a single polynomial solution. No spline LSF representation is used.
Existing spline interpolation of the *wavelength solution* remains part of the
pipeline's coordinate mapping; this does not change the LSF polynomial model.

The apStar supplies its exact wavelength grid, ``VREL`` and chip ``NRES``.
``VREL`` is the visit-to-rest-frame shift; barycentric ``VRAD`` is not
interchangeable with it. The native calibration override currently supports a
single contributing visit and checks the fiber when that metadata is available.

Without ``native_calibration``, the converter instead propagates the three
stored apVisit coefficient arrays and fits a common polynomial to those
model-derived profiles. Such a fit tests consistency with existing solutions;
it cannot establish whether the existing solutions fit the measured data.
The ``fit_h2=True`` option also frees H2 in this output approximation when
checking its degeneracy with width. The default keeps it fixed to zero.

The supplied example produced one cubic GH8 coefficient vector with maximum
held-out kernel errors of about 0.4--0.5 percent of the profile peak after
refitting the native data. These are approximation errors against the propagated
reference, not measured abundance uncertainties. Residual structure and centroid
differences remain and require scientific validation.

Format and coverage
-------------------

All output coordinates are zero-based pixels on the original full apStar grid.
Parameters use the standard ascending-power APOGEE/Doppler polynomial layout,
with ``Xoffset=-origin`` and one output-pixel integration width. A centered,
scaled Chebyshev polynomial basis is used internally for numerical conditioning;
export converts it to ordinary powers. No new wavelength-variation format is
required by a packed-polynomial reader.

The FITS file contains a single ``PARAMETERS`` vector, wavelengths, supported
``DOMAINS``, diagnostic ``PROFILEn`` tables and provenance. Profile tables denote
wavelength regions, not independent coefficient sets. Unsupported chip gaps
and unvalidated edge regions return NaN rows. The reader also accepts the early
three-chip diagnostic files; those need a joint refit to obtain one coefficient
vector. The abandoned spline prototype is not an active supported format.

Kernels describe point-source responses as a function of output-minus-input
pixel. A convolution routine must preserve this orientation for asymmetric
profiles. They already include native pixel integration and resampling; do not
apply a second pixel response. Negative sinc lobes are retained.

Both the native NPZ file and output FITS file preserve provenance and protect
existing files unless overwrite is requested. Input spectra and installed
source checkouts are never modified by these functions. Output is marked
experimental; the package does not automatically replace survey calibrations.

Existing calibration audit
--------------------------

For calibration 40030031, the published parameter arrays match the supplied
visit after the documented factor-of-two transformation. Their headers record
``LSFMETHD=2`` (independent chip fits). The saved blue-chip fiber-22 fit has a
poorly constrained wing around 168 native pixels; the final FITS parameters
match the original fiber-23 solution. The original calibration code has both
nearest-good-fit and adjacent-fiber replacement safeguards.

``aplsf.pro`` also contains ``fitmethod=3`` for fitting all chips jointly.
That older branch is not equivalent to the newer per-chip sky-template path:
the latter explicitly handles sky-line doublets and wings. It should be audited
or generalized rather than merely switched on without validation.

In the supplied measured-line experiment, a shared cubic model trained on 23
lines predicts seven held-out lines about as well as independent chip models.
This supports a common polynomial solution for this test, but does not establish
calibration accuracy across every fiber, exposure or epoch. The line-window
uncertainties also leave residual chi-square above unity.

Doppler normalization compatibility
-----------------------------------

The native reference evaluator follows the original APOGEE IDL convention.
A separate Doppler patch removes a duplicate normalization, includes the highest
integration term omitted by exclusive Python loop bounds, and fixes
``nowings``. The new converter uses an independently checked evaluator and a
lightweight packed-format adapter, so it does not depend on the unpatched
Doppler evaluator or load Doppler's stellar models. The corrected evaluator
matches the original calibration FITS profile images at float32 precision.
Doppler's high-level ``anyarray`` interface also clips negative values, so use
the corrected low-level evaluator when retaining sinc lobes.

Command-line examples are in ``examples/fit_calibration_lsf.py`` and
``examples/derive_apstar_lsf.py``. Unit tests cover integration, analytic
variable-projection derivatives, synthetic recovery, packed coefficient
roundtrips, actual-file resampling and disjoint training/validation pixels.
