"""Build APOGEE calibration products, optionally including dependencies (v3).

``makecal`` is the orchestration layer. Dependency discovery lives in
``dependencies.py`` and each registered builder below creates exactly one
product, assuming that its prerequisites have already been built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import os
from typing import Any, Callable, Mapping

import numpy as np

from ..mkcal import getcal, getnums, readcal
from .dependencies import (
    CalibrationDependencyResolver,
    CalibrationGraph,
)

__all__ = [
    "BUILDERS", "BuilderSpec", "CalibrationContext",
    "calibration_builder", "makecal",
]

BuilderFunction = Callable[[str, "CalibrationContext"], None]


@dataclass(frozen=True)
class BuilderSpec:
    """Registered implementation and data-model root for one product type."""

    caltype: str
    root: str
    function: BuilderFunction


BUILDERS: dict[str, BuilderSpec] = {}


def calibration_builder(caltype: str, root: str):
    """Register a function that builds one calibration product."""

    kind = caltype.lower()

    def register(function: BuilderFunction) -> BuilderFunction:
        if kind in BUILDERS:
            raise ValueError(f"A builder is already registered for {kind!r}")
        BUILDERS[kind] = BuilderSpec(kind, root, function)
        return function

    return register


@dataclass
class CalibrationContext:
    """Shared, typed state used by calibration builders."""

    load: Any
    calfile: str
    allcaldict: Mapping[str, np.ndarray | None]
    clobber: bool = False
    unlock: bool = False
    verbose: bool = False
    librarypsf: bool = False
    psfid: Any = None
    modelpsf: Any = None
    nofit: bool = False
    full: bool = False
    newwave: bool = False
    doplot: bool = False
    dependencies: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def apred(self) -> str:
        return self.load.apred

    @property
    def telescope(self) -> str:
        return self.load.telescope

    def option(self, name: str, default=None):
        """Return a named standard or extra builder option."""

        if hasattr(self, name):
            return getattr(self, name)
        return self.extra.get(name, default)

    def dependency_options(self) -> dict[str, Any]:
        """Return options that can change dependency selection."""

        options = dict(self.extra)
        options.update({
            "librarypsf": self.librarypsf,
            "psfid": self.psfid,
            "modelpsf": self.modelpsf,
        })
        return options

    @staticmethod
    def _same_name(left, right) -> bool:
        left, right = str(left).strip(), str(right).strip()
        try:
            return int(left) == int(right)
        except ValueError:
            return left == right

    def row(self, caltype: str, name, *, required: bool = True):
        """Return the last matching master-calibration table row."""

        table = self.allcaldict.get(caltype)
        if table is not None and table.dtype.names and "name" in table.dtype.names:
            matches = [
                row for row in table
                if self._same_name(row["name"], name)
            ]
            if matches:
                return matches[-1]
        if required:
            raise KeyError(f"No {caltype!r} calibration entry named {name!r}")
        return None

    def frames(self, caltype: str, name, field_name: str = "frames") -> list[int]:
        """Return exposure IDs stored in a calibration-table field."""

        frames = getnums(str(self.row(caltype, name)[field_name]))
        if not frames:
            raise ValueError(f"{caltype}:{name} has no frames in {field_name!r}")
        return frames

    def mjd(self, exposure) -> int:
        return int(self.load.cmjd(int(np.atleast_1d(exposure)[0])))

    def calibrations(self, mjd: int) -> dict[str, Any]:
        return getcal(self.calfile, int(mjd))

    @staticmethod
    def calid(calibrations: Mapping[str, Any], *names: str, default=None):
        """Return the first available spelling of a calibration ID."""

        for name in names:
            value = calibrations.get(name)
            if value not in (None, 0, "0", ""):
                return value
        return default


def _routine(name: str) -> Callable:
    """Load a translated calibration routine only when its builder runs."""

    module_name = f"{__package__}.{name}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise NotImplementedError(
                f"The Python calibration routine {name!r} has not been translated yet"
            ) from error
        raise
    try:
        return getattr(module, name)
    except AttributeError as error:
        raise NotImplementedError(
            f"{module_name!r} does not define {name!r}"
        ) from error


def _context_from_arguments(apred, telescope, options) -> CalibrationContext:
    """Construct a context while preserving the legacy ``makecal`` API."""

    values = dict(options)
    load = values.pop("load", None)
    if load is None:
        if apred is None or telescope is None:
            raise ValueError("Provide either load or both apred and telescope")
        from ...utils.apload import ApLoad

        load = ApLoad(apred=apred, telescope=telescope)

    calfile = values.pop("calfile", None)
    if not calfile:
        drp_dir = os.environ.get("APOGEE_DRP_DIR")
        if not drp_dir:
            raise EnvironmentError(
                "APOGEE_DRP_DIR must be set when calfile is not supplied"
            )
        calfile = os.path.join(drp_dir, "data", "cal", load.instrument + ".par")

    allcaldict = values.pop("allcaldict", None)
    if allcaldict is None:
        allcaldict = readcal(calfile)

    standard = {}
    for key, default in (
        ("clobber", False), ("unlock", False), ("verbose", False),
        ("librarypsf", False), ("psfid", None), ("modelpsf", None),
        ("nofit", False), ("full", False), ("newwave", False),
        ("doplot", False), ("dependencies", False),
    ):
        standard[key] = values.pop(key, default)

    return CalibrationContext(
        load=load, calfile=calfile, allcaldict=allcaldict,
        extra=values, **standard,
    )


def _product_exists(context: CalibrationContext, node) -> bool:
    kind = "det" if node.caltype == "detector" else node.caltype
    spec = BUILDERS.get(kind)
    if spec is None:
        raise ValueError(f"No builder is registered for {node.caltype!r}")
    return bool(context.load.exists(spec.root, num=node.name))


def _run_calibration_graph(
    name, caltype: str, context: CalibrationContext
) -> CalibrationGraph:
    """Resolve and execute a calibration dependency graph."""

    resolver = CalibrationDependencyResolver(
        context.allcaldict,
        context.load,
        exists=(None if context.clobber else
                lambda node: _product_exists(context, node)),
        options=context.dependency_options(),
    )
    graph = resolver.resolve([(caltype, name)])
    for level in graph.topological_levels(missing_only=True):
        for node in level:
            _run_node(node, context)
    return graph


def _run_node(node, context: CalibrationContext) -> None:
    """Dispatch one resolved node to its registered builder."""

    kind = "det" if node.caltype == "detector" else node.caltype
    spec = BUILDERS.get(kind)
    if spec is None:
        raise ValueError(f"No builder is registered for {node.caltype!r}")
    if context.verbose:
        print(f"makecal {kind}: {node.name}")
    spec.function(node.name, context)


def _run_requested_calibration(
    name, caltype: str, context: CalibrationContext
) -> CalibrationGraph:
    """Build only the requested product, without resolving prerequisites."""

    node = CalibrationDependencyResolver.node(caltype, name)
    if node is None:
        raise ValueError(f"Invalid calibration name {name!r}")
    graph = CalibrationGraph(roots={node}, dependencies={node: set()})
    if not context.clobber and _product_exists(context, node):
        graph.existing.add(node)
        return graph
    _run_node(node, context)
    return graph


def makecal(name, caltype, apred=None, telescope=None, **options):
    """Build one calibration, optionally including missing prerequisites.

    The public call remains compatible with the initial translation: callers
    may provide either ``apred`` plus ``telescope`` or an existing ``load``,
    along with ``calfile``, ``allcaldict``, and execution options. By default
    only the requested product is built. Set ``dependencies=True`` to resolve
    and build its complete prerequisite graph first.

    Returns
    -------
    CalibrationGraph
        Resolved graph, including products that already existed.
    """

    kind = caltype.lower()
    if kind == "detector":
        kind = "det"
    if kind not in BUILDERS:
        supported = ", ".join(sorted(BUILDERS))
        raise ValueError(f"Unsupported calibration type {caltype!r}; use {supported}")
    context = _context_from_arguments(apred, telescope, options)
    if context.dependencies:
        return _run_calibration_graph(name, kind, context)
    return _run_requested_calibration(name, kind, context)


@calibration_builder("det", "Detector")
def det(name: str, context: CalibrationContext) -> None:
    row = context.row("det", name)
    _routine("mkdet")(
        name, linid=row["linid"], apred=context.apred,
        telescope=context.telescope, clobber=context.clobber,
        unlock=context.unlock,
    )


@calibration_builder("dark", "Dark")
def dark(name: str, context: CalibrationContext) -> None:
    _routine("mkdark")(
        context.frames("dark", name), apred=context.apred,
        telescope=context.telescope, clobber=context.clobber,
        unlock=context.unlock, verbose=context.verbose,
    )


@calibration_builder("flat", "Flat")
def flat(name: str, context: CalibrationContext) -> None:
    row = context.row("flat", name)
    frames = context.frames("flat", name)
    calibrations = context.calibrations(context.mjd(frames[0]))
    _routine("mkflat")(
        frames, apred=context.apred, telescope=context.telescope,
        darkid=calibrations.get("darkid"), nrep=row["nrep"],
        dithered=bool(row["dithered"]), clobber=context.clobber,
        unlock=context.unlock, verbose=context.verbose,
    )


@calibration_builder("bpm", "BPM")
def bpm(name: str, context: CalibrationContext) -> None:
    row = context.row("bpm", name)
    _routine("mkbpm")(
        name, apred=context.apred, telescope=context.telescope,
        darkid=row["darkid"], flatid=row["flatid"],
        clobber=context.clobber, unlock=context.unlock,
    )


@calibration_builder("fiber", "Fiber")
def fiber(name: str, context: CalibrationContext) -> None:
    calibrations = context.calibrations(context.mjd(name))
    _routine("mkfiber")(
        name, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), clobber=context.clobber,
        unlock=context.unlock,
    )


@calibration_builder("sparse", "Sparse")
def sparse(name: str, context: CalibrationContext) -> None:
    row = context.row("sparse", name)
    frames = context.frames("sparse", name)
    calibrations = context.calibrations(context.mjd(frames[0]))
    maxread = getnums(str(row["maxread"]))
    if len(maxread) != 3:
        raise ValueError(f"sparse:{name} maxread must contain three values")
    _routine("mkepsf")(
        frames, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"),
        fiberid=calibrations.get("fiberid"),
        darkims=getnums(str(row["darkframes"])), dmax=row["dmax"],
        maxread=maxread, clobber=context.clobber, filter=True,
        thresh=0.2, scat=2, unlock=context.unlock,
    )


@calibration_builder("littrow", "Littrow")
def littrow(name: str, context: CalibrationContext) -> None:
    cmjd = context.load.cmjd(int(name))
    calibrations = context.calibrations(int(cmjd))
    _routine("mklittrow")(
        name, cmjd=cmjd, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"),
        sparseid=calibrations.get("sparseid"),
        fiberid=calibrations.get("fiberid"), clobber=context.clobber,
        unlock=context.unlock,
    )


@calibration_builder("psf", "PSF")
def psf(name: str, context: CalibrationContext) -> None:
    calibrations = context.calibrations(context.mjd(name))
    _routine("mkpsf")(
        name, bpmid=calibrations.get("bpmid"),
        darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"),
        sparseid=calibrations.get("sparseid"),
        fiberid=calibrations.get("fiberid"),
        littrowid=calibrations.get("littrowid"),
        clobber=context.clobber, unlock=context.unlock,
    )


@calibration_builder("modelpsf", "PSFModel")
def modelpsf(name: str, context: CalibrationContext) -> None:
    row = context.row("modelpsf", name)
    _routine("mkmodelpsf")(
        name, sparseid=row["sparse"], psfid=row["psf"],
        clobber=context.clobber, unlock=context.unlock,
    )


def _selected_psf(context: CalibrationContext, calibrations):
    if context.psfid is not None:
        return context.psfid, None
    if context.librarypsf:
        return None, None
    model = context.modelpsf
    if model is None:
        model = context.calid(calibrations, "modelpsf", "modelpsfid")
    return None, model


@calibration_builder("fpi", "WaveFPI")
def fpi(name: str, context: CalibrationContext) -> None:
    calibrations = context.calibrations(context.mjd(name))
    psfid, model = _selected_psf(context, calibrations)
    _routine("mkfpi")(
        name, name=name, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=psfid,
        modelpsf=model, fiberid=calibrations.get("fiberid"),
        clobber=context.clobber, unlock=context.unlock,
        psflibrary=context.librarypsf,
    )


@calibration_builder("persist", "Persist")
def persist(name: str, context: CalibrationContext) -> None:
    row = context.row("persist", name)
    cmjd = context.load.cmjd(int(name))
    calibrations = context.calibrations(int(cmjd))
    _routine("mkpersist")(
        name, row["darkid"], row["flatid"], apred=context.apred,
        telescope=context.telescope, thresh=row["thresh"], cmjd=cmjd,
        darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"),
        sparseid=calibrations.get("sparseid"),
        fiberid=calibrations.get("fiberid"), clobber=context.clobber,
        unlock=context.unlock,
    )


@calibration_builder("persistmodel", "PersistModel")
def persistmodel(name: str, context: CalibrationContext) -> None:
    context.row("persistmodel", name)
    _routine("mkpersistmodel")(name)


@calibration_builder("flux", "Flux")
def flux(name: str, context: CalibrationContext) -> None:
    calibrations = context.calibrations(context.mjd(name))
    psfid, model = _selected_psf(context, calibrations)
    _routine("mkflux")(
        [int(name)], cmjd=context.load.cmjd(int(name)),
        darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=psfid,
        modelpsf=model, littrowid=calibrations.get("littrowid"),
        waveid=calibrations.get("waveid"), clobber=context.clobber,
        unlock=context.unlock,
    )


@calibration_builder("response", "Response")
def response(name: str, context: CalibrationContext) -> None:
    row = context.row("response", name)
    calibrations = context.calibrations(context.mjd(name))
    _routine("mkflux")(
        [int(name)], cmjd=context.load.cmjd(int(name)),
        darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=row["psf"],
        littrowid=calibrations.get("littrowid"),
        waveid=calibrations.get("waveid"), temp=row["temp"],
        clobber=context.clobber, unlock=context.unlock,
    )


@calibration_builder("wave", "Wave")
def wave(name: str, context: CalibrationContext) -> None:
    row = context.row("wave", name, required=False)
    if row is None:
        frames, output_name, row_psfid = [int(name)], str(name), None
    else:
        frames = getnums(str(row["frames"]))
        output_name, row_psfid = str(row["name"]), row["psfid"]
    calibrations = context.calibrations(context.mjd(frames[0]))
    if row_psfid not in (None, 0, "0", ""):
        psfid, model = row_psfid, None
    else:
        psfid, model = _selected_psf(context, calibrations)
    _routine("mkwave")(
        frames, apred=context.apred, telescope=context.telescope,
        name=output_name, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=psfid,
        modelpsf=model, fiberid=calibrations.get("fiberid"),
        clobber=context.clobber, nofit=context.nofit,
        unlock=context.unlock, plot=context.doplot,
    )


@calibration_builder("multiwave", "Wave")
def multiwave(name: str, context: CalibrationContext) -> None:
    _routine("mkmultiwave")(
        context.frames("multiwave", name), name=name,
        calfile=context.calfile, clobber=context.clobber,
        unlock=context.unlock, psflibrary=context.librarypsf,
    )


@calibration_builder("dailywave", "Wave")
def dailywave(name: str, context: CalibrationContext) -> None:
    mjd = int(name)
    calibrations = context.calibrations(mjd)
    psfid, model = _selected_psf(context, calibrations)
    _routine("mkdailywave")(
        mjd, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=psfid,
        modelpsf=model, fiberid=calibrations.get("fiberid"),
        clobber=context.clobber, nofit=context.nofit,
        unlock=context.unlock, psflibrary=context.librarypsf,
    )


@calibration_builder("telluric", "Telluric")
def telluric(name: str, context: CalibrationContext) -> None:
    _routine("mktelluric")(
        name, clobber=context.clobber, unlock=context.unlock,
    )


@calibration_builder("lsf", "LSF")
def lsf(name: str, context: CalibrationContext) -> None:
    row = context.row("lsf", name)
    frames = context.frames("lsf", name)
    calibrations = context.calibrations(context.mjd(frames[0]))
    _routine("mklsf")(
        frames, context.calid(calibrations, "multiwaveid", "waveid"),
        darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=row["psfid"],
        fiberid=calibrations.get("fiberid"), full=context.full,
        newwave=context.newwave, clobber=context.clobber,
        pl=context.doplot, unlock=context.unlock,
    )
