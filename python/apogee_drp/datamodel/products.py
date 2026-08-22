"""Physical-file definitions for logical APOGEE calibration products.

``sdss-tree`` remains responsible for constructing an individual pathname.
This module describes which individual files together constitute a complete
logical calibration product.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple


APOGEE_CHIPS = ("a", "b", "c")

MJDMode = Literal["exposure", "mjd", "none"]
Resolver = Callable[[object, object, Optional[int]], List[str]]


@dataclass(frozen=True)
class FileComponent:
    """One sdss-tree product belonging to a logical APOGEE product."""

    root: str
    chips: Optional[Tuple[str, ...]] = APOGEE_CHIPS


@dataclass(frozen=True)
class ProductSpec:
    """Definition of the files required for one logical product."""

    components: Tuple[FileComponent, ...] = ()
    mjd_mode: MJDMode = "exposure"
    require_nonempty: bool = True
    resolver: Optional[Resolver] = None


def _as_list(value) -> List[str]:
    """Normalize the scalar/dictionary result from ``ApLoad.filename``."""
    if isinstance(value, dict):
        return [str(filename) for filename in value.values()]
    return [str(value)]


def _standard_files(load, spec: ProductSpec, name, mjd) -> List[str]:
    files = []
    for component in spec.components:
        files.extend(_as_list(load.filename(
            component.root,
            num=name,
            mjd=mjd,
            chip=component.chips,
        )))
    return files


def _summary_files(root: str) -> Resolver:
    """Resolve three chip FITS files and the builder's chipless .tab file."""

    def resolve(load, name, mjd):
        files = _as_list(load.filename(
            root, num=name, mjd=mjd, chip=APOGEE_CHIPS))
        template = Path(load.filename(root, num=name, mjd=mjd))
        return files + [str(template.with_suffix(".tab"))]

    return resolve

def _dailywave_files(load, name, mjd):
    """Resolve daily-wave files whose identifier is an unpadded MJD."""
    identifier = str(int(name))
    files = _as_list(load.filename("Wave", num=name, mjd=mjd,
                                   chip=APOGEE_CHIPS))
    resolved = []
    for filename in files:
        path = Path(filename)
        prefix, separator, _ = path.stem.rpartition("-")
        if not separator:
            raise ValueError(
                f"Cannot replace Wave identifier in filename: {filename}"
            )
        resolved.append(str(path.with_name(f"{prefix}-{identifier}{path.suffix}")))
    return resolved

def _lsf_files(load, name, mjd):
    files = _as_list(load.filename(
        "LSF", num=name, mjd=mjd, chip=APOGEE_CHIPS))
    template = Path(load.filename("LSF", num=name, mjd=mjd))
    diagnostic = template.with_name(
        f"{template.stem}-diagnostics{template.suffix}")
    return files + [str(diagnostic)]


def _telluric_files(load, name, mjd):
    """Resolve the temporary compound-ID Telluric filename layout."""
    identifier = str(name).strip()
    parts = identifier.split("-")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"Telluric ID must be '<waveid>-<lsfid>', got {name!r}")
    if any(int(part) <= 0 for part in parts):
        raise ValueError("Telluric waveid and lsfid must be positive")
    directory = Path(load.filename("Telluric", num=0, directory=True))
    prefix = getattr(
        load, "prefix", "ap" if "apo" in load.telescope else "as")
    return [
        str(directory / f"{prefix}Telluric-{chip}-{identifier}.fits")
        for chip in APOGEE_CHIPS
    ]


PRODUCTS: Dict[str, ProductSpec] = {
    "detector": ProductSpec((FileComponent("Detector"),)),
    "dark": ProductSpec(resolver=_summary_files("Dark")),
    "flat": ProductSpec(resolver=_summary_files("Flat")),
    "bpm": ProductSpec((FileComponent("BPM"),)),
    "fiber": ProductSpec((FileComponent("Fiber"),)),
    "sparse": ProductSpec((
        FileComponent("Sparse", chips=None),
        FileComponent("EPSF"),
    )),
    "littrow": ProductSpec((FileComponent("Littrow", chips=("b",)),)),
    "psf": ProductSpec((
        FileComponent("PSF"),
        FileComponent("EPSF"),
        FileComponent("ETrace"),
    )),
    "modelpsf": ProductSpec((FileComponent("PSFModel"),)),
    "fpi": ProductSpec((FileComponent("WaveFPI"),)),
    "persist": ProductSpec((FileComponent("Persist"),)),
    "persistmodel": ProductSpec((FileComponent("PersistModel"),)),
    "flux": ProductSpec((FileComponent("Flux"),)),
    "response": ProductSpec((FileComponent("Response"),)),
    "wave": ProductSpec((FileComponent("Wave"),)),
    "multiwave": ProductSpec((FileComponent("Wave"),)),
    "dailywave": ProductSpec(mjd_mode="mjd",
                             resolver=_dailywave_files),
    "telluric": ProductSpec(mjd_mode="none", resolver=_telluric_files),
    "lsf": ProductSpec(resolver=_lsf_files),
}


def product_spec(product: str) -> ProductSpec:
    """Return the registered specification for a logical product."""
    key = str(product).strip().lower()
    try:
        return PRODUCTS[key]
    except KeyError as error:
        supported = ", ".join(sorted(PRODUCTS))
        raise ValueError(
            f"Unknown APOGEE product {product!r}; expected one of: {supported}"
        ) from error


def product_mjd(load, product: str, name) -> Optional[int]:
    """Resolve the directory MJD prescribed by a product specification."""
    mode = product_spec(product).mjd_mode
    if mode == "none":
        return None
    if mode == "mjd":
        return int(name)
    first_identifier = str(name).strip().split("-", 1)[0]
    if not first_identifier:
        raise ValueError(f"Product {product!r} has an empty identifier")
    return int(load.cmjd(int(first_identifier)))


def product_files(load, product: str, name, *, mjd=None) -> List[str]:
    """Return every physical file required for a logical product.

    The returned order is stable: components follow their registry order and
    detector chips follow ``a``, ``b``, ``c`` unless specified otherwise.
    """
    spec = product_spec(product)
    resolved_mjd = product_mjd(load, product, name) if mjd is None else int(mjd)
    if spec.resolver is not None:
        files = spec.resolver(load, name, resolved_mjd)
    else:
        files = _standard_files(load, spec, name, resolved_mjd)
    if not files:
        raise RuntimeError(f"Product {product!r} resolved to no files")
    return [str(filename) for filename in files]


def file_is_complete(filename, *, require_nonempty=True) -> bool:
    """Return whether one physical product file is complete."""
    path = Path(filename)
    return path.is_file() and (not require_nonempty or path.stat().st_size > 0)


def product_status(load, product: str, name, *, mjd=None) -> Dict[str, bool]:
    """Return completeness keyed by every required physical filename."""
    spec = product_spec(product)
    return {
        filename: file_is_complete(
            filename, require_nonempty=spec.require_nonempty)
        for filename in product_files(load, product, name, mjd=mjd)
    }


def product_exists(load, product: str, name, *, mjd=None) -> bool:
    """Return whether all physical files for a logical product are complete."""
    return all(product_status(load, product, name, mjd=mjd).values())

def product_delete(load, product, name, *, mjd=None, missing_ok=True,
                   dry_run=False, verbose=False):
    """Delete every physical file belonging to a logical APOGEE product.

    Parameters
    ----------
    load : ApLoad
        APOGEE file loader.
    product : str
        Logical product name, such as ``"dark"`` or ``"psf"``.
    name
        Product identifier.
    mjd : int, optional
        Explicit product MJD. If omitted, it is resolved from the product
        specification.
    missing_ok : bool, optional
        If True, silently ignore files that do not exist.
    dry_run : bool, optional
        If True, report which existing files would be deleted without
        removing them.
    verbose : bool, optional
        Print each file that is deleted, or would be deleted in dry-run mode.

    Returns
    -------
    list of str
        Files deleted, or files that would be deleted when ``dry_run=True``.

    Raises
    ------
    FileNotFoundError
        If a required file is missing and ``missing_ok=False``.
    IsADirectoryError
        If a resolved product path is unexpectedly a directory.
    """
    # Preserve order while protecting against accidental duplicate entries.
    filenames = list(dict.fromkeys(
        product_files(load, product, name, mjd=mjd)
    ))

    # Validate all targets before deleting anything.
    for filename in filenames:
        path = Path(filename)

        # A symlink can safely be unlinked even if it points to a directory.
        if path.exists() and path.is_dir() and not path.is_symlink():
            raise IsADirectoryError(
                f"Refusing to delete product directory: {path}"
            )

        if not os.path.lexists(path) and not missing_ok:
            raise FileNotFoundError(
                f"Required product file does not exist: {path}"
            )

    deleted = []

    for filename in filenames:
        path = Path(filename)

        if not os.path.lexists(path):
            continue

        if dry_run:
            if verbose:
                print(f"Would delete: {path}")
        else:
            path.unlink()
            if verbose:
                print(f"Deleted: {path}")

        deleted.append(str(path))

    return deleted


__all__ = [
    "APOGEE_CHIPS", "FileComponent", "PRODUCTS", "ProductSpec",
    "file_is_complete", "product_exists", "product_files", "product_mjd",
    "product_spec", "product_status", "product_delete",
]
