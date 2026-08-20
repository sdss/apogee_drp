"""Build APOGEE calibration products, optionally including dependencies (v9).

``makecal`` is the orchestration layer. Dependency discovery lives in
``dependencies.py`` and each registered builder below creates exactly one
product, assuming that its prerequisites have already been built.

1) the builder registry (BuilderSpec)
2) calibration context (CalibrationContext)
     this object contains everything the builders need
     use this instead of **kw
3) context initialization
4) calibration routine loading
5) existing product handling
6) executing one node
7) requested-product-only execution
8) dependency execution
9) the public makecal() function
10) the individual builders
    det,dark,flat,bpm,fiber,sparse,littrow,psf,modelpsf,
    fpi,persist,persistmodel,flux,response,wave,multiwave,
    dailywave,telluric,lsf
11) psf selection

The central idea is that makecal() decides what to run,
dependencies.py decides what is required, and each builder knows only
how to construct its own calibration product.

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
from .detector import build_detector
from .dark import build_dark
from .flat import build_flat
from .bpm import build_bpm
from .psfcal import build_fiber, build_modelpsf, build_psf, build_sparse
from .lsf import build_lsf
from .telluric import build_telluric
from .littrow import build_littrow
from .fpi import build_fpi
from .wavecal import build_dailywave, build_multiwave, build_wave
from .persistence import build_persist
from .fluxcal import build_flux, build_response


__all__ = [
    "BUILDERS", "BuilderSpec", "CalibrationContext",
    "calibration_builder", "makecal",
]

BuilderFunction = Callable[[str, "CalibrationContext"], None]


@dataclass(frozen=True)
class BuilderSpec:
    """Registered implementation and data-model root for one product type."""
    caltype: str
    function: BuilderFunction


BUILDERS: dict[str, BuilderSpec] = {}


def calibration_builder(caltype: str):
    """Register a function that builds one calibration product."""
    kind = caltype.lower()

    def register(function: BuilderFunction) -> BuilderFunction:
        if kind in BUILDERS:
            raise ValueError(f"A builder is already registered for {kind!r}")

        BUILDERS[kind] = BuilderSpec(
            caltype=kind,
            function=function,
        )
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
    return_graph: bool = False
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
        ("return_graph", False),
    ):
        standard[key] = values.pop(key, default)

    return CalibrationContext(
        load=load, calfile=calfile, allcaldict=allcaldict,
        extra=values, **standard,
    )


def _product_exists(context: CalibrationContext, node) -> bool:
    """Return whether the complete logical calibration product exists."""
    return context.load.product_exists(
        node.caltype,
        node.name,
    )

def _report_existing(context, node):
    if context.verbose:
        filenames = context.load.product_files(node.caltype, node.name)
        print(
            f" {node.caltype} product already made: "
            f"{len(filenames)} files"
        )

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
    for node in sorted(graph.existing):
        _report_existing(context, node)
    for level in graph.topological_levels(missing_only=True):
        for node in level:
            _run_node(node, context)
    return graph


def _run_node(node, context: CalibrationContext) -> None:
    """Dispatch one resolved node to its registered builder."""

    kind = node.caltype
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
        _report_existing(context, node)
        return graph
    _run_node(node, context)
    return graph


def makecal(caltype, name, apred=None, telescope=None, **options):
    """Build one calibration, optionally including missing prerequisites.

    Parameters
    ----------
    caltype : str
        Calibration type, such as ``detector``, ``flat``, or ``dailywave``.
    name
        Calibration ID or compound product name.
    apred, telescope : str, optional
        Reduction version and telescope. They are required unless ``load`` is
        supplied. By default only the requested product is built; set
        ``dependencies=True`` to build its complete prerequisite graph first.

    By default the function returns ``None``, matching ``makecal.pro``. Set
    ``return_graph=True`` to return the resolved :class:`CalibrationGraph`.
    """

    kind = caltype.lower()
    if kind not in BUILDERS:
        supported = ", ".join(sorted(BUILDERS))
        raise ValueError(f"Unsupported calibration type {caltype!r}; use {supported}")
    context = _context_from_arguments(apred, telescope, options)
    if context.dependencies:
        graph = _run_calibration_graph(name, kind, context)
    else:
        graph = _run_requested_calibration(name, kind, context)
    return graph if context.return_graph else None


@calibration_builder("detector")
def detector(name: str, context: CalibrationContext) -> None:
    row = context.row("det", name)
    build_detector(
        name, linid=row["linid"], apred=context.apred,
        telescope=context.telescope, clobber=context.clobber,
        unlock=context.unlock, verbose=context.verbose,
    )


@calibration_builder("dark")
def dark(name: str, context: CalibrationContext) -> None:
    build_dark(
        context.frames("dark", name), apred=context.apred,
        telescope=context.telescope, clobber=context.clobber,
        unlock=context.unlock, verbose=context.verbose,
    )


@calibration_builder("flat")
def flat(name: str, context: CalibrationContext) -> None:
    row = context.row("flat", name)
    frames = context.frames("flat", name)
    calibrations = context.calibrations(context.mjd(frames[0]))
    build_flat(
        frames, apred=context.apred, telescope=context.telescope,
        darkid=calibrations.get("darkid"), nrep=row["nrep"],
        dithered=bool(row["dithered"]), clobber=context.clobber,
        unlock=context.unlock, verbose=context.verbose,
    )


@calibration_builder("bpm")
def bpm(name: str, context: CalibrationContext) -> None:
    row = context.row("bpm", name)
    build_bpm(
        name, apred=context.apred, telescope=context.telescope,
        darkid=row["darkid"], flatid=row["flatid"],
        clobber=context.clobber, unlock=context.unlock,
        verbose=context.verbose,
    )


@calibration_builder("fiber")
def fiber(name: str, context: CalibrationContext) -> None:
    calibrations = context.calibrations(context.mjd(name))
    build_fiber(
        name, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), bpmid=calibrations.get("bpmid"),
        apred=context.apred, telescope=context.telescope,
        clobber=context.clobber, unlock=context.unlock,
        verbose=context.verbose,
    )


@calibration_builder("sparse")
def sparse(name: str, context: CalibrationContext) -> None:
    row = context.row("sparse", name)
    frames = context.frames("sparse", name)
    calibrations = context.calibrations(context.mjd(frames[0]))
    maxread = getnums(str(row["maxread"]))
    if len(maxread) != 3:
        raise ValueError(f"sparse:{name} maxread must contain three values")
    build_sparse(
        frames, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"),
        bpmid=calibrations.get("bpmid"),
        fiberid=calibrations.get("fiberid"),
        darkframes=getnums(str(row["darkframes"])), dmax=row["dmax"],
        maxread=maxread, apred=context.apred, telescope=context.telescope,
        clobber=context.clobber, threshold=0.2, unlock=context.unlock,
        verbose=context.verbose,
    )


@calibration_builder("littrow")
def littrow(name: str, context: CalibrationContext) -> None:
    cmjd = context.load.cmjd(int(name))
    calibrations = context.calibrations(int(cmjd))
    build_littrow(
        name, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"),
        bpmid=calibrations.get("bpmid"),
        sparseid=calibrations.get("sparseid"),
        fiberid=calibrations.get("fiberid"), apred=context.apred,
        telescope=context.telescope, clobber=context.clobber,
        unlock=context.unlock, verbose=context.verbose,
    )


@calibration_builder("psf")
def psf(name: str, context: CalibrationContext) -> None:
    calibrations = context.calibrations(context.mjd(name))
    build_psf(
        name, bpmid=calibrations.get("bpmid"),
        darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"),
        sparseid=calibrations.get("sparseid"),
        fiberid=calibrations.get("fiberid"),
        littrowid=calibrations.get("littrowid"),
        apred=context.apred, telescope=context.telescope,
        clobber=context.clobber, unlock=context.unlock,
        verbose=context.verbose,
    )

    
@calibration_builder("modelpsf")
def modelpsf(name: str, context: CalibrationContext) -> None:
    row = context.row("modelpsf", name)
    build_modelpsf(
        name, sparseid=row["sparse"], psfid=row["psf"],
        apred=context.apred, telescope=context.telescope,
        clobber=context.clobber, unlock=context.unlock,
        verbose=context.verbose,
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


@calibration_builder("fpi")
def fpi(name: str, context: CalibrationContext) -> None:
    calibrations = context.calibrations(context.mjd(name))
    psfid, model = _selected_psf(context, calibrations)
    build_fpi(
        name, name=name, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=psfid,
        modelpsf=model, fiberid=calibrations.get("fiberid"),
        apred=context.apred, telescope=context.telescope,
        clobber=context.clobber, unlock=context.unlock,
        librarypsf=context.librarypsf, verbose=context.verbose,
    )


@calibration_builder("persist")
def persist(name: str, context: CalibrationContext) -> None:
    row = context.row("persist", name)
    cmjd = context.load.cmjd(int(name))
    calibrations = context.calibrations(int(cmjd))
    build_persist(
        name, row["darkid"], row["flatid"], apred=context.apred,
        telescope=context.telescope, thresh=row["thresh"], cmjd=cmjd,
        darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"),
        sparseid=calibrations.get("sparseid"),
        fiberid=calibrations.get("fiberid"), clobber=context.clobber,
        unlock=context.unlock, verbose=context.verbose,
    )


@calibration_builder("persistmodel")
def persistmodel(name: str, context: CalibrationContext) -> None:
    context.row("persistmodel", name)
    _routine("mkpersistmodel")(name)


@calibration_builder("flux")
def flux(name: str, context: CalibrationContext) -> None:
    calibrations = context.calibrations(context.mjd(name))
    psfid, model = _selected_psf(context, calibrations)
    build_flux(
        [int(name)], apred=context.apred, telescope=context.telescope,
        cmjd=context.load.cmjd(int(name)),
        darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=psfid,
        modelpsf=model, littrowid=calibrations.get("littrowid"),
        waveid=calibrations.get("waveid"),
        persistid=calibrations.get("persistid"), clobber=context.clobber,
        unlock=context.unlock, verbose=context.verbose,
    )


@calibration_builder("response")
def response(name: str, context: CalibrationContext) -> None:
    row = context.row("response", name)
    calibrations = context.calibrations(context.mjd(name))
    build_response(
        int(name), waveid=calibrations.get("waveid"), temp=row["temp"],
        apred=context.apred, telescope=context.telescope,
        clobber=context.clobber, unlock=context.unlock,
        verbose=context.verbose,
    )


@calibration_builder("wave")
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
    build_wave(
        frames, apred=context.apred, telescope=context.telescope,
        name=output_name, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=psfid,
        modelpsf=model, fiberid=calibrations.get("fiberid"),
        clobber=context.clobber, nofit=context.nofit,
        unlock=context.unlock, plot=context.doplot, verbose=context.verbose,
    )


@calibration_builder("multiwave")
def multiwave(name: str, context: CalibrationContext) -> None:
    build_multiwave(
        context.frames("multiwave", name), name=name,
        apred=context.apred, telescope=context.telescope,
        clobber=context.clobber, unlock=context.unlock,
        dependencies=False, verbose=context.verbose,
    )


@calibration_builder("dailywave")
def dailywave(name: str, context: CalibrationContext) -> None:
    mjd = int(name)
    calibrations = context.calibrations(mjd)
    psfid, model = _selected_psf(context, calibrations)
    build_dailywave(
        mjd, darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=psfid,
        modelpsf=model, fiberid=calibrations.get("fiberid"),
        clobber=context.clobber, nofit=context.nofit,
        unlock=context.unlock, librarypsf=context.librarypsf,
        dependencies=context.dependencies, apred=context.apred,
        telescope=context.telescope, verbose=context.verbose,
    )

    
@calibration_builder("telluric")
def telluric(name: str, context: CalibrationContext) -> None:
    build_telluric(
        name, apred=context.apred, telescope=context.telescope,
        clobber=context.clobber, unlock=context.unlock,
        verbose=context.verbose,
    )

    
@calibration_builder("lsf")
def lsf(name: str, context: CalibrationContext) -> None:
    row = context.row("lsf", name)
    frames = context.frames("lsf", name)
    calibrations = context.calibrations(context.mjd(frames[0]))
    build_lsf(
        frames, context.calid(calibrations, "multiwaveid", "waveid"),
        darkid=calibrations.get("darkid"),
        flatid=calibrations.get("flatid"), psfid=row["psfid"],
        fiberid=calibrations.get("fiberid"), full=context.full,
        newwave=context.newwave, clobber=context.clobber,
        plot=context.doplot, unlock=context.unlock, apred=context.apred,
        telescope=context.telescope, verbose=context.verbose,
    )
