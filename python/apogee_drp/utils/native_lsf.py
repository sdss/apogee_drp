"""Global native LSF calibration products and legacy per-chip projections.

Each physical fiber has its own shared native-pixel polynomial, fitted to all
three detectors. Standard per-chip pixel vectors are exact algebraic exports.
The earlier wavelength-based model remains available as an explicit comparison
mode, with its compatibility projection errors recorded.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import tempfile

import numpy as np
from astropy.io import fits
from astropy.table import Table
from numpy.polynomial import chebyshev as cheb
from scipy.interpolate import CubicSpline

from .calibration_lsf import (
    LOGW0,
    DLOGW,
    NativePolynomialLSF,
    load_calibration_windows,
    fit_native_polynomial,
    evaluate_held_out,
    edge_profile,
)
from .lsf_parameters import pack_parameters
from .apstar_lsf import evaluate_apogee_gh
from .pixel_lsf import NativePixelLSF, fit_native_pixel


class NativeLSFUnavailable(ValueError):
    """The calibration explicitly has no valid solution for this fiber."""


def default_line_catalog():
    """Locate the DRP airglow catalog; an explicit catalog can always be supplied."""
    roots = [Path(__file__).resolve().parents[3]]
    if os.environ.get("APOGEE_DRP_DIR"):
        roots.insert(0, Path(os.environ["APOGEE_DRP_DIR"]))
    for root in roots:
        path = root / "data/skylines/airglow.txt"
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Supply line_catalog, or set APOGEE_DRP_DIR to the DRP checkout"
    )


def _atomic_fits(hdul, filename, overwrite):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        hdul.writeto(temporary, overwrite=True, checksum=True)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication, including a competing writer.
            os.link(temporary, path)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _native_profiles(model, wavelength, centers, offsets):
    forward = CubicSpline(np.arange(len(wavelength)), np.log10(wavelength))
    q = (forward(centers) - LOGW0) / DLOGW
    left = (forward(centers[:, None] + offsets - 0.5) - LOGW0) / DLOGW
    right = (forward(centers[:, None] + offsets + 0.5) - LOGW0) / DLOGW
    return edge_profile(
        np.minimum(left, right),
        np.maximum(left, right),
        q,
        model.fields(q),
        model.horder,
    )


def project_native_parameters(model, wavelength, degree=3, halfwidth=16):
    """Project the shared solution into a classic native-pixel GH vector.

    Uses the local wavelength derivative to transform widths and reverses odd
    Hermite terms on decreasing wavelength axes. This is an approximation to
    the nonlinear coordinate transform, not another fit to calibration data.
    Returns packed parameters and an independent dense profile error check.
    """
    wave = np.asarray(wavelength, float)
    if wave.ndim != 1 or not np.isfinite(wave).all() or np.any(wave <= 0):
        raise ValueError("Invalid native wavelength array")
    delta = np.diff(wave)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError("Native wavelengths must be strictly monotonic")
    n = len(wave)
    origin, scale = (n - 1) / 2, max((n - 1) / 2, 1)
    centers = np.linspace(0, n - 1, 160)
    forward = CubicSpline(np.arange(n), np.log10(wave))
    slope = forward(centers, 1) / DLOGW
    f = model.fields((forward(centers) - LOGW0) / DLOGW)
    local = np.zeros((model.horder + 3, len(centers)))
    local[0] = f[0] / abs(slope)
    for m in range(3, model.horder + 1):
        local[m] = f[m - 2] * np.sign(slope) ** m
    local[-2] = f[-2]
    local[-1] = f[-1] / abs(slope)
    coefficients = cheb.chebfit((centers - origin) / scale, local.T, degree).T
    power = np.zeros_like(coefficients)
    for i, row in enumerate(coefficients):
        converted = cheb.cheb2poly(row)
        power[i, : len(converted)] = converted
    power /= scale ** np.arange(degree + 1)[None, :]
    orders = np.array([np.flatnonzero(row)[-1] if np.any(row) else 0 for row in power])
    packed = pack_parameters(
        dict(
            binsize=1.0,
            Xoffset=-origin,
            Horder=model.horder,
            Porder=orders[:-2],
            GHcoefs=power[:-2],
            Wproftype=1,
            nWpar=2,
            WPorder=orders[-2:],
            Wcoefs=power[-2:],
        )
    )
    test = np.linspace(0.37, n - 1 - 0.37, 257)
    offsets = np.arange(-halfwidth, halfwidth + 1)
    reference = _native_profiles(model, wave, test, offsets)
    approximation = evaluate_apogee_gh(offsets, test, packed)
    rel = np.max(abs(approximation - reference), axis=1) / np.max(reference, axis=1)
    return packed, float(rel.max())


@dataclass
class NativeLSFCalibration:
    """A set of per-fiber global native solutions and explicit failure records.

    Fiber IDs are physical APOGEE IDs 1..300; array row is always 300-fiber.
    Filled solutions retain their donor fibers, method and original failure.
    """

    models: dict
    status: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        files,
        *,
        saved_fit=None,
        line_catalog=None,
        fibers=None,
        degree=3,
        horder=5,
        min_snr=15.0,
        halfwidth=16,
        on_error="record",
        logger=None,
        coordinate="pixel",
        degree_overrides=None,
        fill_missing=None,
        fill_method="local_linear",
        fill_neighbors=5,
        check_filled_peaks=True,
        sigma_options=None,
    ):
        """Fit three wavelength-calibrated ap1D files, independently per fiber.

        By default fits all 300 fibers and records failures. Use on_error='raise'
        for an all-or-nothing fit. The DRP airglow catalog avoids any dependency
        on an old IDL fit or .sav file. saved_fit is an optional comparison input.
        Failed native-pixel fits are filled from measured neighboring fibers
        by default, with explicit provenance. fill_missing=False keeps them
        unavailable. Unrequested fibers remain NaN. The retained wavelength
        mode defaults to leaving failed fibers unavailable. on_error='raise'
        aborts before filling. degree_overrides selects per-fiber degrees.
        """
        if coordinate not in ("pixel", "log_wavelength"):
            raise ValueError("coordinate must be pixel or log_wavelength")
        if len(files) != 3 or on_error not in ("record", "raise"):
            raise ValueError("Supply three ap1D files and on_error='record' or 'raise'")
        fibers = list(range(1, 301)) if fibers is None else [int(f) for f in fibers]
        if (
            not fibers
            or len(set(fibers)) != len(fibers)
            or any(f < 1 or f > 300 for f in fibers)
        ):
            raise ValueError("fibers must be unique IDs in 1..300")
        if saved_fit is None:
            line_catalog = (
                default_line_catalog() if line_catalog is None else line_catalog
            )
            if isinstance(line_catalog, (str, Path)):
                line_catalog = Table.read(line_catalog, format="ascii.commented_header")
        elif isinstance(saved_fit, (str, Path)):
            from scipy.io import readsav

            saved_fit = readsav(saved_fit, python_dict=True)["linestr"]
        overrides = {} if degree_overrides is None else dict(degree_overrides)
        degree_overrides = {}
        for fiber_id, value in overrides.items():
            if (
                int(fiber_id) != float(fiber_id)
                or not 1 <= int(fiber_id) <= 300
                or int(value) != value
                or value < 0
            ):
                raise ValueError("Invalid per-fiber polynomial degree override")
            degree_overrides[int(fiber_id)] = int(value)
        models, status = {}, {}
        for fiber in fibers:
            try:
                lines = load_calibration_windows(
                    files,
                    saved_fit,
                    fiber=fiber,
                    halfwidth=halfwidth,
                    min_snr=min_snr,
                    line_catalog=line_catalog,
                )
                fiber_degree = degree_overrides.get(fiber, degree)
                if coordinate == "pixel":
                    model, lines = fit_native_pixel(
                        lines, files, fiber, degree=fiber_degree, horder=horder
                    )
                else:
                    model = fit_native_polynomial(
                        lines, degree=fiber_degree, horder=horder
                    )
                validation, _, _ = evaluate_held_out(model, lines)
                model.diagnostics.update(
                    validation=validation, input_provenance=lines.provenance
                )
                if not np.isfinite(model.coefficients).all():
                    raise ValueError("Nonfinite fitted coefficients")
                models[fiber] = model
                status[fiber] = "OK"
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                if on_error == "raise":
                    raise RuntimeError(f"Native LSF fiber {fiber}: {exc}") from exc
                status[fiber] = str(exc)
            if logger:
                logger.info("Native LSF fiber %s: %s", fiber, status[fiber])
        if not models:
            raise RuntimeError("No successful native LSF fibers; " + str(status))
        result = cls(
            models,
            status,
            dict(
                files=[str(f) for f in files],
                degree=degree,
                horder=horder,
                coordinate=coordinate,
                degree_overrides=degree_overrides,
                fiber_mapping="row=300-fiber",
            ),
        )
        should_fill = (
            coordinate == "pixel" if fill_missing is None else bool(fill_missing)
        )
        if should_fill:
            sigma_settings=dict(line_catalog=line_catalog,saved_fit=saved_fit,
                                min_snr=min_snr,halfwidth=halfwidth)
            if sigma_options:sigma_settings.update(sigma_options)
            result.fill_missing(
                [f for f in fibers if f not in models],
                method=fill_method,
                neighbors=fill_neighbors,
                files=files,
                check_peaks=check_filled_peaks,
                sigma_options=sigma_settings,
            )
        return result

    @property
    def filled_fibers(self):
        """Physical fiber IDs estimated from measured neighbors."""
        return sorted(
            f for f, m in self.models.items() if m.diagnostics.get("filled", False)
        )

    def fill_missing(
        self, targets=None, *, method="local_linear", neighbors=5, degree=1, files=None,
        check_peaks=True, sigma_options=None
    ):
        """Estimate failed fibers, preserving provenance and all measured models.

        The default fits a local line to coefficient variation over five nearby
        measured fibers. Stored polynomials are first put on a common pixel
        basis. 'linear' instead interpolates between the nearest bracketing pair;
        'nearest' copies one fiber. 'polynomial' uses the specified local degree.
        Edges use the nearest donor rather than extrapolating. Filled fibers
        never become donors, including across repeated calls. Only missing
        targets are modified. With files supplied, valid measured chip geometry
        is used; invalid/unavailable geometry is estimated from the donors.

        By default all absent fibers are filled when this method is explicitly
        called. fit() passes only the failed requested fibers, so a subset fit
        does not silently create unrequested fibers. Failed fallback attempts
        raise instead of assigning an invalid profile.
        """
        from .fiber_lsf import interpolate_fiber_model

        if method not in ("nearest", "linear", "local_linear", "polynomial"):
            raise ValueError("Unknown fiber filling method")
        if (
            int(neighbors) != neighbors
            or neighbors < 1
            or int(degree) != degree
            or degree < 0
        ):
            raise ValueError("Invalid fiber filling neighbor count or degree")
        targets = list(range(1, 301)) if targets is None else [int(f) for f in targets]
        if any(not 1 <= f <= 300 for f in targets):
            raise ValueError("Fiber targets must be in 1..300")
        targets = sorted(set(f for f in targets if f not in self.models))
        if not targets:
            return self
        donors = {
            f: m
            for f, m in self.models.items()
            if not m.diagnostics.get("filled", False)
        }
        if not donors:
            raise ValueError("No measured LSFs available to fill missing fibers")
        if not all(isinstance(m, NativePixelLSF) for m in donors.values()):
            raise TypeError("Fiber filling requires native pixel models")
        inputs = files if files is not None else self.provenance.get("files", [])
        geometry = {}
        if len(inputs) == 3 and all(Path(p).is_file() for p in inputs):
            for target in targets:
                offsets = []
                try:
                    for path in inputs:
                        with fits.open(path) as h:
                            wave = np.asarray(h[4].data[300 - target], float)
                            if (
                                not np.isfinite(wave).all()
                                or np.any(wave <= 0)
                                or not (
                                    np.all(np.diff(wave) > 0)
                                    or np.all(np.diff(wave) < 0)
                                )
                            ):
                                raise ValueError("Invalid target wavelength solution")
                            coeff = np.asarray(h[5].data)
                            value = (
                                coeff[0, 300 - target]
                                if coeff.shape == (14, 300)
                                else coeff[300 - target, 0]
                            )
                            if not np.isfinite(value):
                                raise ValueError("Invalid chip offset")
                            offsets.append(float(value))
                    geometry[target] = np.array(offsets)
                except (ValueError, IndexError, KeyError):
                    pass
        staged = {}
        for target in targets:
            failure = self.status.get(target, "not fitted")
            attempted = []
            for candidate in dict.fromkeys([method, "linear", "nearest"]):
                try:
                    model = interpolate_fiber_model(
                        donors,
                        target,
                        candidate,
                        neighbors,
                        degree,
                        chip_offsets=geometry.get(target),
                        original_failure=failure,
                    )
                    if attempted:
                        model.diagnostics["fill_fallback_reasons"] = attempted
                    staged[target] = model
                    break
                except ValueError as exc:
                    attempted.append(str(exc))
            if target not in staged:
                raise NativeLSFUnavailable(
                    f"Could not fill fiber {target}: {attempted}"
                )
        self.models.update(staged)
        for target, model in staged.items():
            self.status[target] = "FILLED_" + model.diagnostics["fill_method"].upper()
        self.provenance["fiber_filling"] = dict(
            method=method,
            neighbors=neighbors,
            degree=degree,
            targets=targets,
            measured_donors_only=True,
        )
        if check_peaks and len(inputs)==3 and all(Path(p).is_file() for p in inputs):
            self.check_filled_peaks(inputs,targets=list(staged),**(sigma_options or {}))
        return self

    def check_filled_peaks(self, files, targets=None, *, line_catalog=None,
                           saved_fit=None, min_snr=15., halfwidth=16,
                           force=False, diagnostic_directory=None, **refinement_options):
        """Check estimated fibers against their own peaks and refine sigma only.

        Available clean peaks are used even when fewer than six occur on a chip.
        Sigma's existing polynomial is constrained by training lines; held-out
        lines and physical full-detector bounds gate acceptance. Hermite terms
        and wing parameters remain neighbor estimates. Missing flux, bad line
        selection or failed validation retain the original model and explicit
        status. Both attempted and accepted corrections are stored in each
        model's diagnostics. Repeated checks are a no-op unless force=True.

        Set diagnostic_directory to save per-fiber sigma and peak-fit PDFs.

        This runs automatically after filling when fit/fill_missing have access
        to the native files. Use check_filled_peaks=False in fit(), or
        check_peaks=False in fill_missing(), to retain a purely neighbor-based
        calibration. The calibration remains marked filled even after sigma is
        constrained by data, because its other LSF fields are still estimated.
        """
        from copy import deepcopy
        from .calibration_lsf import BAD_BITS
        from .pixel_lsf import pixel_line_windows
        from .sigma_refinement import refine_sigma
        if len(files)!=3 or not all(Path(p).is_file() for p in files):
            raise ValueError("Supply three existing native ap1D files")
        targets=self.filled_fibers if targets is None else [int(f) for f in targets]
        if saved_fit is None and line_catalog is None:line_catalog=default_line_catalog()
        for fiber in targets:
            original=self.model(fiber)
            if not original.diagnostics.get("filled",False):
                continue
            if original.diagnostics.get("sigma_refinement") and not force:
                continue
            availability=[]
            for path in files:
                with fits.open(path) as h:
                    flux=np.array(h[1].data[300-fiber],float)
                    error=np.array(h[2].data[300-fiber],float)
                    mask=np.array(h[3].data[300-fiber],np.uint32)
                good=np.isfinite(flux)&np.isfinite(error)&(error>0)&(error<1e5)&((mask&BAD_BITS)==0)
                availability.append(dict(good_pixels=int(good.sum()),
                    nonzero_usable_pixels=int(np.count_nonzero(flux[good]))))
            result=deepcopy(original)
            if not any(a['nonzero_usable_pixels'] for a in availability):
                info=dict(status="NO_USABLE_DATA",accepted=False,availability=availability)
                result.diagnostics['sigma_refinement']=info
                result.diagnostics['sigma_data_constrained']=False
            else:
                try:
                    lines=load_calibration_windows(files,saved_fit,fiber=fiber,
                        halfwidth=halfwidth,min_snr=min_snr,line_catalog=line_catalog,
                        min_lines_per_chip=0)
                    pixel,_,_=pixel_line_windows(lines,files,fiber,chip_offsets=original.chip_offsets)
                    result,info=refine_sigma(original,pixel,**refinement_options)
                    info['availability']=availability
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    info=dict(status="PEAK_CHECK_FAILED",accepted=False,reason=str(exc),availability=availability)
                    result.diagnostics['sigma_refinement']=info
                    result.diagnostics['sigma_data_constrained']=False
            info['input_files']=[str(p) for p in files]
            self.models[fiber]=result
        self.provenance['filled_sigma_checks']=dict(targets=targets,fields=['sigma'],
            input_files=[str(p) for p in files])
        if diagnostic_directory is not None:
            self.plot_filled_peaks(files, diagnostic_directory, targets=targets,
                line_catalog=line_catalog, saved_fit=saved_fit,
                min_snr=min_snr, halfwidth=halfwidth)
        return self

    def plot_diagnostics(self, output):
        """Save a run-wide PDF of status, peak residuals and native LSF maps.

        Includes all attempted fibers, validation coverage, sigma and shape
        fields versus fiber and detector position. Can be called after reading
        a saved native-pixel calibration; requires matplotlib, no raw data.
        Returns the output path and does not modify calibration models.
        """
        from .lsf_diagnostics import plot_calibration
        return plot_calibration(self, output)

    def plot_filled_peaks(self, files, output_directory, targets=None, *,
                          line_catalog=None, saved_fit=None, min_snr=15., halfwidth=16):
        """Save one multipage diagnostic PDF per filled fiber and return paths.

        Each PDF shows original/adopted sigma versus native detector pixel,
        locations of training and held-out peaks, and data/model overlays for
        every usable peak. Rejected candidate widths are identified separately.
        No-data fibers receive an explicit status page. Calibration models are
        not modified. Requires matplotlib only when called.
        """
        from .pixel_lsf import pixel_line_windows
        from .lsf_diagnostics import plot_sigma_check
        if saved_fit is None and line_catalog is None:
            line_catalog = default_line_catalog()
        targets = self.filled_fibers if targets is None else targets
        outputs = []
        for fiber in targets:
            model = self.model(fiber)
            lines = None
            if model.diagnostics.get('sigma_refinement', {}).get('status') != 'NO_USABLE_DATA':
                try:
                    raw = load_calibration_windows(files, saved_fit, fiber=fiber,
                        halfwidth=halfwidth, min_snr=min_snr, line_catalog=line_catalog,
                        min_lines_per_chip=0)
                    lines, _, _ = pixel_line_windows(raw, files, fiber,
                        chip_offsets=model.chip_offsets)
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    lines = None
            outputs.append(plot_sigma_check(model, lines,
                Path(output_directory) / f'lsf-fiber-{fiber:03d}.pdf'))
        return outputs

    def model(self, fiber):
        """Return an available measured or explicitly flagged estimated fiber model."""
        fiber = int(fiber)
        if fiber not in self.models:
            raise NativeLSFUnavailable(
                f"No global native LSF for fiber {fiber}: {self.status.get(fiber,'not fitted')}"
            )
        return self.models[fiber]

    def global_hdu(self, fibers=None):
        """Serialize the exact model, with a coordinate convention in the header."""
        selected = sorted(self.models) if fibers is None else list(fibers)
        if not selected:
            if not self.models:
                raise ValueError(
                    "Cannot serialize an empty calibration without a coordinate template"
                )
            template = self.global_hdu([next(iter(self.models))])
            template.data = template.data[:0].copy()
            template.header["LSFFILL"] = False
            template.header["SIGREFIN"] = False
            return template
        stored_size = max(self.model(f).coefficients.size for f in selected)
        rows = []
        for fiber in selected:
            m = self.model(fiber)
            rows.append(
                (
                    fiber,
                    m.horder,
                    m.coefficients.shape[0],
                    m.coefficients.shape[1],
                    m.origin,
                    m.scale,
                    np.pad(
                        m.coefficients.ravel(), (0, stored_size - m.coefficients.size)
                    ),
                    json.dumps(m.diagnostics),
                )
            )
        table = Table(
            rows=rows,
            names=[
                "FIBER",
                "HORDER",
                "NFIELD",
                "NCOEF",
                "ORIGIN",
                "SCALE",
                "COEFFICIENTS",
                "DIAGNOSTICS",
            ],
        )
        pixel = all(isinstance(self.model(f), NativePixelLSF) for f in selected)
        if any(isinstance(self.model(f), NativePixelLSF) for f in selected) != pixel:
            raise ValueError(
                "Do not mix native pixel and wavelength models in one calibration"
            )
        if pixel:
            table["CHIP_OFFSETS"] = np.array(
                [self.model(f).chip_offsets for f in selected]
            )
            table["NPIX"] = np.array([self.model(f).npix for f in selected])
        hdu = fits.BinTableHDU(table, name="GLOBAL_LSF")
        hdu.header["LSFVER"] = 2 if pixel else 1
        hdu.header["SIGREFIN"] = any(self.model(f).diagnostics.get("sigma_data_constrained",False) for f in selected)
        hdu.header["LSFFILL"] = any(
            self.model(f).diagnostics.get("filled", False) for f in selected
        )
        hdu.header["BASIS"] = "CHEBYSHEV"
        hdu.header["COORDSYS"] = "NATIVE_PIXEL" if pixel else "LOG_WAVELENGTH"
        hdu.header["LSFFRAME"] = "DETECTOR" if pixel else "OBSERVED"
        if not pixel:
            hdu.header["LOGW0"] = LOGW0
            hdu.header["DLOGW"] = DLOGW
        return hdu

    @classmethod
    def from_hdul(cls, hdul):
        """Read exact global models from a calibration or apVisit HDUList."""
        h = hdul["GLOBAL_LSF"]
        pixel = (
            h.header.get("LSFVER") == 2
            and h.header.get("COORDSYS") == "NATIVE_PIXEL"
            and h.header.get("LSFFRAME") == "DETECTOR"
        )
        wavelength = (
            h.header.get("LSFVER") == 1
            and h.header.get("LOGW0") == LOGW0
            and h.header.get("DLOGW") == DLOGW
            and h.header.get("LSFFRAME") == "OBSERVED"
        )
        if h.header.get("BASIS") != "CHEBYSHEV" or not (pixel or wavelength):
            raise ValueError("Unknown global native LSF coordinate convention")
        models = {}
        for r in h.data:
            flat = np.array(r["COEFFICIENTS"], float)
            count = int(r["NFIELD"]) * int(r["NCOEF"])
            if count > flat.size or np.any(flat[count:] != 0):
                raise ValueError("Invalid global LSF coefficient count or padding")
            co = flat[:count].reshape(int(r["NFIELD"]), int(r["NCOEF"]))
            if not np.isfinite(co).all() or float(r["SCALE"]) <= 0:
                raise ValueError("Invalid global native LSF coefficients or scale")
            fiber = int(r["FIBER"])
            if fiber in models or not 1 <= fiber <= 300:
                raise ValueError("Invalid or duplicate global LSF fiber")
            args = (
                co,
                float(r["ORIGIN"]),
                float(r["SCALE"]),
                int(r["HORDER"]),
                json.loads(r["DIAGNOSTICS"]),
            )
            if pixel:
                offsets = np.array(r["CHIP_OFFSETS"], float)
                if (
                    offsets.shape != (3,)
                    or not np.isfinite(offsets).all()
                    or int(r["NPIX"]) < 2
                ):
                    raise ValueError("Invalid native pixel coordinate mapping")
                models[fiber] = NativePixelLSF(
                    *args, chip_offsets=offsets, npix=int(r["NPIX"])
                )
            else:
                models[fiber] = NativePolynomialLSF(*args)
        status = {
            f: (
                "FILLED_" + m.diagnostics.get("fill_method", "UNKNOWN").upper()
                if m.diagnostics.get("filled", False)
                else "OK"
            )
            for f, m in models.items()
        }
        if "FIBER_STATUS" in hdul:
            status.update(
                {int(r["FIBER"]): str(r["STATUS"]) for r in hdul["FIBER_STATUS"].data}
            )
        provenance = (
            json.loads(hdul["PROVENANCE"].data["JSON"][0])
            if "PROVENANCE" in hdul
            else {}
        )
        return cls(models, status, provenance)

    @classmethod
    def read(cls, filename):
        with fits.open(filename, checksum=True) as hdul:
            return cls.from_hdul(hdul)

    def write(
        self,
        files,
        outputs,
        *,
        global_output=None,
        overwrite=False,
        halfwidth=16,
        projection_degree=3,
        max_projection_error=0.01,
    ):
        """Write standard apLSF-[abc] files, full profile cubes and exact models.

        Primary arrays are [nparameters,300], first-extension profile cubes are
        [2*halfwidth+1,300,npixel], as in classic apLSF files. Cubes are evaluated
        from the exact global model. Native pixels have binsize=1; the visit
        combiner performs its existing dither-coordinate conversion. Projection
        errors apply only to the retained wavelength model. Native pixel
        models export exactly, so their projection error is zero.
        A combined global_output file is optional. Existing files are preserved
        unless overwrite=True. Products are staged before publication.
        """
        if len(files) != 3 or len(outputs) != 3 or halfwidth < 8:
            raise ValueError("Supply three input/output paths and halfwidth>=8")
        allpaths = list(outputs) + ([] if global_output is None else [global_output])
        if len(set(map(str, allpaths))) != len(allpaths):
            raise ValueError("Output paths must differ")
        for path in allpaths:
            if Path(path).exists() and not overwrite:
                raise FileExistsError(path)
        status_rows = []
        for f in range(1, 301):
            diagnostics = self.models[f].diagnostics if f in self.models else {}
            current = self.status.get(f, "not fitted")
            status_rows.append(
                (
                    f,
                    current,
                    bool(diagnostics.get("filled", False)),
                    str(diagnostics.get("original_failure", current)),
                    ",".join(map(str, diagnostics.get("source_fibers", []))),
                    bool(diagnostics.get("sigma_data_constrained",False)),
                    diagnostics.get("sigma_refinement",{}).get("status",
                        "NOT_CHECKED" if diagnostics.get("filled",False) else "DIRECT_FIT"),
                )
            )
        status = Table(
            rows=status_rows,
            names=["FIBER", "STATUS", "FILLED", "ORIGINAL_STATUS", "SOURCES", "SIGMA_REFINED", "PEAK_CHECK"],
        )
        status_hdu = fits.BinTableHDU(status, name="FIBER_STATUS")
        offsets = np.arange(-halfwidth, halfwidth + 1)
        prepared = []
        try:
            for chip, (source, target) in enumerate(zip(files, outputs)):
                with fits.open(source, memmap=True) as hdul:
                    header = hdul[0].header.copy()
                    wavelengths = np.array(hdul[4].data, float)
                if wavelengths.shape[0] != 300:
                    raise ValueError("Expected 300 native fiber rows")
                npix = wavelengths.shape[1]
                projections = {}
                errors = {}
                cube = np.full((len(offsets), 300, npix), np.nan, dtype=np.float32)
                for fiber, m in self.models.items():
                    row = 300 - fiber
                    wave = wavelengths[row]
                    if isinstance(m, NativePixelLSF):
                        if m.npix != npix:
                            raise ValueError(
                                "Native pixel model and detector size differ"
                            )
                        packed, error = m.parameters(chip), 0.0
                    else:
                        packed, error = project_native_parameters(
                            m, wave, projection_degree, halfwidth
                        )
                    if error > max_projection_error:
                        raise ValueError(
                            f"Native compatibility projection chip {chip}, fiber {fiber}: {error:.3%} exceeds {max_projection_error:.3%}"
                        )
                    projections[fiber] = packed
                    errors[fiber] = error
                    if isinstance(m, NativePixelLSF):
                        cube[:, row] = m.native_profiles(
                            np.arange(npix), offsets, chip
                        ).T
                    else:
                        cube[:, row] = _native_profiles(
                            m, wave, np.arange(npix), offsets
                        ).T
                npar = max(map(len, projections.values()))
                pars = np.full((npar, 300), np.nan)
                for f, v in projections.items():
                    pars[:, 300 - f] = 0
                    pars[: len(v), 300 - f] = v
                header["LSFMETHD"] = (3, "Global fit to all three chips per fiber")
                header["LSFVER"] = 1
                header["LSFCOORD"] = "PIXEL"
                header["LSFGLOB"] = True
                header["EXTNAME"] = "LSFPARS"
                header["BINSIZE"] = 1.0
                pixel_model = all(
                    isinstance(m, NativePixelLSF) for m in self.models.values()
                )
                header["PIXEXACT"] = pixel_model
                header.add_history(
                    "GLOBAL_LSF contains the shared per-fiber native model."
                )
                header.add_history(
                    "Primary parameters are exact algebraic pixel exports."
                    if pixel_model
                    else "Primary parameters are compatibility projections of the wavelength model."
                )
                primary = fits.PrimaryHDU(pars, header=header)
                image = fits.ImageHDU(cube, name="LSF ARRAY")
                image.header["CRVAL3"] = -halfwidth
                image.header["CRPIX3"] = 1
                image.header["CDELT3"] = 1
                image.header["BINSIZE"] = 1.0
                quality = fits.BinTableHDU(
                    Table(
                        rows=sorted(errors.items()), names=["FIBER", "MAX_PEAK_ERROR"]
                    ),
                    name="PROJECTION_QA",
                )
                product = fits.HDUList(
                    [primary, image, self.global_hdu(), status_hdu.copy(), quality]
                )
                # Stage all three products before replacing any existing product.
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                fd, temp = tempfile.mkstemp(
                    prefix=Path(target).name + ".stage.", dir=Path(target).parent
                )
                os.close(fd)
                prepared.append((temp, target))
                product.writeto(temp, overwrite=True, checksum=True)
                del cube, product
            if global_output is not None:
                head = fits.Header()
                head["LSFMETHD"] = 3
                head["LSFVER"] = 1
                primary = fits.PrimaryHDU(header=head)
                prov = fits.BinTableHDU(
                    Table(rows=[(json.dumps(self.provenance),)], names=["JSON"]),
                    name="PROVENANCE",
                )
                _atomic_fits(
                    fits.HDUList([primary, self.global_hdu(), status_hdu.copy(), prov]),
                    global_output,
                    overwrite,
                )
            for temp, target in prepared:
                if overwrite:
                    os.replace(temp, target)
                else:
                    os.link(temp, target)
                    os.unlink(temp)
        finally:
            for temp, _ in prepared:
                if os.path.exists(temp):
                    os.unlink(temp)
        return [str(p) for p in allpaths]


def visit_global_lsf_hdu(chip_files, fiber):
    """Carry the exact per-fiber global model into a new apVisit product.

    Classic calibration files without GLOBAL_LSF remain supported. All chips
    must agree on the exact shared solution; mixed calibrations are rejected.
    """
    return _visit_global_lsf_hdu(chip_files, fiber)


from functools import lru_cache


@lru_cache(maxsize=12)
def _cached_calibration(path, modified_ns, size):
    with fits.open(path, memmap=True) as h:
        return NativeLSFCalibration.from_hdul(h) if "GLOBAL_LSF" in h else None


def _visit_global_lsf_hdu(chip_files, fiber):
    if not any(chip_files):
        return None
    if len(chip_files) != 3 or not all(chip_files):
        raise ValueError("All three chip calibration paths are required")
    calibrations = []
    for path in chip_files:
        stat = Path(path).stat()
        calibrations.append(
            _cached_calibration(
                str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size
            )
        )
    if all(c is None for c in calibrations):
        return None
    if any(c is None for c in calibrations):
        raise ValueError("Cannot mix global and classic chip LSF calibrations")
    available = [int(fiber) in c.models for c in calibrations]
    if not any(available):
        hdu = calibrations[0].global_hdu()
        hdu.data = hdu.data[:0].copy()
        hdu.header["LSFFILL"] = False
        hdu.header["SIGREFIN"] = False
        hdu.header["LSFSTAT"] = "NO_LSF"
        hdu.header["FIBERID"] = int(fiber)
        return hdu
    if not all(available):
        raise ValueError("Chip calibrations disagree on fiber LSF availability")
    models = [c.model(fiber) for c in calibrations]
    first = models[0]
    for other in models[1:]:
        if (
            type(first) is not type(other)
            or first.horder != other.horder
            or (
                isinstance(first, NativePixelLSF)
                and (
                    first.npix != other.npix
                    or not np.array_equal(first.chip_offsets, other.chip_offsets)
                )
            )
            or first.origin != other.origin
            or first.scale != other.scale
            or not np.array_equal(first.coefficients, other.coefficients)
        ):
            raise ValueError(
                "Chip calibrations do not share the same global native LSF"
            )
    return calibrations[0].global_hdu([int(fiber)])
