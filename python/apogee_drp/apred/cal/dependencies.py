"""Plan APOGEE calibration dependencies before starting worker jobs (v3).

The calibration builders historically call ``makecal`` recursively.  That is
convenient interactively, but it means dependencies outside a requested MJD
range are discovered serially inside a Slurm job.  This module computes the
complete graph up front without creating any products.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from ..mkcal import getnums

__all__ = [
    "CalibrationNode",
    "CalibrationGraph",
    "CalibrationDependencyResolver",
    "apload_exists",
    "calibration_dependency_tree",
]


DEFAULT_MASTER_CALTYPES = (
    "detector",
    "dark",
    "flat",
    "bpm",
    "fiber",
    "sparse",
    "littrow",
    "response",
    "modelpsf",
    "multiwave",
    "lsf",
)


def _canonical_name(value) -> str:
    """Normalize numeric IDs while preserving compound names."""

    text = str(value).strip()
    try:
        return str(int(text))
    except ValueError:
        return text


_APLOAD_ROOTS = {
    "detector": "Detector",
    "dark": "Dark",
    "flat": "Flat",
    "bpm": "BPM",
    "fiber": "Fiber",
    "sparse": "Sparse",
    "littrow": "Littrow",
    "modelpsf": "PSFModel",
    "psf": "PSF",
    "wave": "Wave",
    "multiwave": "Wave",
    "lsf": "LSF",
    "persist": "Persist",
    "response": "Response",
}


def apload_exists(load, node: "CalibrationNode") -> bool:
    """Check a graph node using the corresponding ``ApLoad.exists`` root."""

    root = _APLOAD_ROOTS.get(node.caltype)
    if root is None:
        raise ValueError(f"No ApLoad root is defined for {node.caltype!r}")
    return bool(load.exists(root, num=node.name))


def calibration_dependency_tree(
    mjdstart: int,
    mjdstop: int | None = None,
    *,
    apred: str | None = None,
    telescope: str | None = None,
    load=None,
    caltypes: Sequence[str] | None = None,
    calfile: str | None = None,
    check_exists: bool = True,
    print_tree: bool = True,
    logger=None,
) -> "CalibrationGraph":
    """Plan every calibration needed for an inclusive MJD range.

    This is the public convenience interface for this module.  It selects
    master calibrations whose validity intervals overlap the requested MJDs,
    recursively adds prerequisites outside that range, optionally checks the
    reduction tree for products that already exist, and returns a dependency
    graph whose topological levels can be submitted in parallel.

    Parameters
    ----------
    mjdstart, mjdstop
        Inclusive MJD range.  If ``mjdstop`` is omitted, plan one MJD.
    apred, telescope
        Reduction version and telescope.  Required unless ``load`` is given.
    load
        Existing :class:`~apogee_drp.utils.apload.ApLoad` instance.
    caltypes
        Root master-calibration types.  The default matches
        :func:`mkmastercals`.
    calfile
        Master calibration index.  The instrument index under
        ``$APOGEE_DRP_DIR/data/cal`` is used by default.
    check_exists
        Mark complete products already present in the reduction tree and
        exclude them from ``graph.missing``.
    print_tree
        Print the resolved tree before returning it.
    logger : logging.Logger, optional
        Logger used for dependency-tree output. If not supplied, output is
        printed to the terminal.


    Returns
    -------
    graph : CalibrationGraph
        Complete dependency graph.  Use
        ``graph.topological_levels(missing_only=True)`` for execution groups.

    Examples
    --------
    >>> graph = calibration_dependency_tree(
    ...     60000, 60100, apred="1.6", telescope="apo25m"
    ... )
    >>> levels = graph.topological_levels(missing_only=True)
    """

    if mjdstop is None:
        mjdstop = mjdstart
    mjdstart, mjdstop = int(mjdstart), int(mjdstop)
    if mjdstop < mjdstart:
        raise ValueError("mjdstop must be greater than or equal to mjdstart")

    if load is None:
        if apred is None or telescope is None:
            raise ValueError("Provide either load or both apred and telescope")
        from ...utils.apload import ApLoad

        load = ApLoad(apred=apred, telescope=telescope)

    if calfile is None:
        drp_dir = os.environ.get("APOGEE_DRP_DIR")
        if not drp_dir:
            raise EnvironmentError(
                "APOGEE_DRP_DIR must be set when calfile is not supplied"
            )
        calfile = os.path.join(drp_dir, "data", "cal", load.instrument + ".par")

    from ..mkcal import readcal

    caldict = readcal(calfile)
    exists = (lambda node: apload_exists(load, node)) if check_exists else None
    resolver = CalibrationDependencyResolver(caldict, load, exists=exists)
    requested_types = DEFAULT_MASTER_CALTYPES if caltypes is None else caltypes
    mjds = np.arange(mjdstart, mjdstop + 1, dtype=int)
    roots = resolver.roots_for_mjds(mjds, requested_types)
    graph = resolver.resolve(roots)
    if print_tree:
        tree = graph.format_tree(show_existing=check_exists)
        summary = (
            f'{len(graph.roots)} requested products; '
            f'{len(graph.nodes)} including dependencies; '
            f'{len(graph.missing)} missing.'
        )

        if logger is not None:
            logger.info('Calibration dependency tree:')
            for line in tree.splitlines():
                logger.info(line)
            logger.info(summary)
        else:
            print('Calibration dependency tree:')
            print(tree)
            print()
            print(summary)

    return graph


@dataclass(frozen=True, order=True)
class CalibrationNode:
    """One calibration product, uniquely identified by type and name."""

    caltype: str
    name: str

    def __str__(self) -> str:
        return f"{self.caltype}:{self.name}"


@dataclass
class CalibrationGraph:
    """Dependency graph returned by :class:`CalibrationDependencyResolver`."""

    roots: set[CalibrationNode] = field(default_factory=set)
    dependencies: dict[CalibrationNode, set[CalibrationNode]] = field(
        default_factory=dict
    )
    existing: set[CalibrationNode] = field(default_factory=set)

    @property
    def nodes(self) -> set[CalibrationNode]:
        result = set(self.dependencies)
        for values in self.dependencies.values():
            result.update(values)
        return result

    @property
    def missing(self) -> set[CalibrationNode]:
        return self.nodes - self.existing

    def required_by_type(self, missing_only: bool = True) -> dict[str, list[str]]:
        """Return calibration names grouped for ``mkmastercals`` stages."""

        selected = self.missing if missing_only else self.nodes
        result: dict[str, list[str]] = {}
        for node in sorted(selected):
            result.setdefault(node.caltype, []).append(node.name)
        return result

    def required_names(self, caltype: str, missing_only: bool = True) -> set[str]:
        """Return names needed in one calibration stage."""

        selected = self.missing if missing_only else self.nodes
        return {
            node.name for node in selected
            if node.caltype == caltype.lower()
        }

    def is_required(
        self, caltype: str, name, missing_only: bool = True
    ) -> bool:
        """Return whether one product should be run by ``mkmastercals``."""

        return _canonical_name(name) in self.required_names(
            caltype, missing_only=missing_only
        )

    def topological_levels(self, missing_only: bool = False) -> list[list[CalibrationNode]]:
        """Return parallelizable levels, with prerequisites first."""

        selected = self.missing if missing_only else self.nodes
        remaining = set(selected)
        levels: list[list[CalibrationNode]] = []
        while remaining:
            ready = sorted(
                node for node in remaining
                if not ((self.dependencies.get(node, set()) & selected) & remaining)
            )
            if not ready:
                cycle = ", ".join(str(node) for node in sorted(remaining))
                raise ValueError(f"Calibration dependency cycle detected among: {cycle}")
            levels.append(ready)
            remaining.difference_update(ready)
        return levels

    def format_tree(self, show_existing: bool = True) -> str:
        """Format roots and their recursive prerequisites as readable text."""

        lines: list[str] = []

        def visit(node: CalibrationNode, prefix: str, path: set[CalibrationNode]) -> None:
            state = " [exists]" if node in self.existing else " [missing]"
            lines.append(prefix + str(node) + (state if show_existing else ""))
            if node in path:
                lines[-1] += " [cycle]"
                return
            children = sorted(self.dependencies.get(node, set()))
            for index, child in enumerate(children):
                last = index == len(children) - 1
                visit(child, prefix + ("    " if last else "|   "), path | {node})

        for index, root in enumerate(sorted(self.roots)):
            if index:
                lines.append("")
            visit(root, "", set())
        return "\n".join(lines)


class CalibrationDependencyResolver:
    """Resolve dependencies encoded by the APOGEE calibration builders.

    Parameters
    ----------
    caldict
        Dictionary returned by :func:`apogee_drp.apred.mkcal.readcal`.
    load
        :class:`~apogee_drp.utils.apload.ApLoad` instance.  Its ``cmjd``
        method is used to obtain the observing MJD of exposure IDs.
    exists
        Optional ``exists(node)`` callback.  It should return ``True`` only
        when the complete calibration product is already available.
    """

    def __init__(
        self,
        caldict: Mapping[str, np.ndarray | None],
        load,
        exists: Callable[[CalibrationNode], bool] | None = None,
    ) -> None:
        self.caldict = caldict
        self.load = load
        self.exists = exists

    @staticmethod
    def node(caltype: str, name) -> CalibrationNode | None:
        if name is None:
            return None
        text = _canonical_name(name)
        if text in ("", "0", "None"):
            return None
        kind = caltype.lower()
        if kind == "det":
            kind = "detector"
        return CalibrationNode(kind, text)

    @staticmethod
    def _same_name(left, right) -> bool:
        left, right = str(left).strip(), str(right).strip()
        try:
            return int(left) == int(right)
        except ValueError:
            return left == right

    def _row(self, caltype: str, name, required: bool = True):
        table = self.caldict.get(caltype)
        if table is not None and "name" in table.dtype.names:
            matches = [row for row in table if self._same_name(row["name"], name)]
            if matches:
                return matches[-1]
        if required:
            raise KeyError(f"No {caltype!r} calibration entry named {name!r}")
        return None

    def _valid(self, caltype: str, mjd: int) -> CalibrationNode | None:
        table = self.caldict.get(caltype)
        if table is None:
            return None
        matches = table[(mjd >= table["mjd1"]) & (mjd <= table["mjd2"])]
        if len(matches) == 0:
            return None
        return self.node(caltype, matches[-1]["name"])

    def _mjd(self, exposure) -> int:
        return int(self.load.cmjd(int(exposure)))

    def _row_mjd(self, caltype: str, name, frame_field: str | None = None) -> int:
        if frame_field is None:
            return self._mjd(name)
        row = self._row(caltype, name)
        frames = getnums(str(row[frame_field]))
        if not frames:
            raise ValueError(f"{caltype}:{name} has no frames in {frame_field}")
        return self._mjd(frames[0])

    @staticmethod
    def _clean(nodes: Iterable[CalibrationNode | None]) -> set[CalibrationNode]:
        return {node for node in nodes if node is not None}

    def direct_dependencies(self, node: CalibrationNode) -> set[CalibrationNode]:
        """Return the immediate prerequisites of one calibration product."""

        kind, name = node.caltype, node.name
        if kind == "detector":
            return set()
        if kind == "dark":
            mjd = self._row_mjd("dark", name, "frames")
            return self._clean([self._valid("det", mjd)])
        if kind == "flat":
            mjd = self._row_mjd("flat", name, "frames")
            return self._clean([self._valid("dark", mjd)])
        if kind == "bpm":
            row = self._row("bpm", name)
            return self._clean([
                self.node("dark", row["darkid"]),
                self.node("flat", row["flatid"]),
            ])
        if kind == "fiber":
            # Fiber is now an ETrace-only product; sparse is deliberately not
            # a dependency, which avoids the historical fiber/sparse cycle.
            mjd = self._row_mjd("fiber", name)
            return self._clean([
                self._valid("dark", mjd), self._valid("flat", mjd),
            ])
        if kind == "sparse":
            mjd = self._row_mjd("sparse", name, "frames")
            return self._clean([
                self._valid("dark", mjd), self._valid("flat", mjd),
                self._valid("bpm", mjd), self._valid("fiber", mjd),
            ])
        if kind in ("littrow", "psf"):
            mjd = self._row_mjd("littrow", name) if kind == "littrow" else self._mjd(name)
            dependencies = [
                self._valid("dark", mjd), self._valid("flat", mjd),
                self._valid("sparse", mjd), self._valid("fiber", mjd),
            ]
            if kind == "psf":
                dependencies.append(self._valid("littrow", mjd))
            return self._clean(dependencies)
        if kind == "modelpsf":
            row = self._row("modelpsf", name)
            return self._clean([
                self.node("sparse", row["sparse"]),
                self.node("psf", row["psf"]),
            ])
        if kind == "wave":
            row = self._row("wave", name, required=False)
            if row is None:
                mjd = self._mjd(name)
                psf = self._valid("modelpsf", mjd)
            else:
                frames = getnums(str(row["frames"]))
                mjd = self._mjd(frames[0])
                psf = self.node("psf", row["psfid"])
            return self._clean([
                self._valid("bpm", mjd), self._valid("fiber", mjd), psf,
            ])
        if kind == "multiwave":
            row = self._row("multiwave", name)
            frames = getnums(str(row["frames"])) or []
            # MKMULTIWAVE calls MAKECAL,WAVE on every other frame; adjacent
            # entries are normally the paired arc exposures for one solution.
            return self._clean(self.node("wave", frame) for frame in frames[::2])
        if kind == "lsf":
            row = self._row("lsf", name)
            frames = getnums(str(row["frames"]))
            mjd = self._mjd(frames[0])
            return self._clean([
                self._valid("multiwave", mjd),
                self.node("psf", row["psfid"]),
            ])
        if kind == "persist":
            mjd = self._row_mjd("persist", name)
            return self._clean([
                self._valid("dark", mjd), self._valid("flat", mjd),
            ])
        if kind == "response":
            row = self._row("response", name)
            mjd = self._mjd(name)
            return self._clean([
                self.node("psf", row["psfid"]), self._valid("wave", mjd),
                self._valid("fiber", mjd), self._valid("littrow", mjd),
            ])
        raise ValueError(f"No dependency rule is defined for calibration type {kind!r}")

    def roots_for_mjds(
        self, mjds: Sequence[int], caltypes: Sequence[str]
    ) -> set[CalibrationNode]:
        """Select the products whose validity ranges overlap input MJDs."""

        values = np.asarray(mjds, dtype=int)
        roots: set[CalibrationNode] = set()
        aliases = {"detector": "det", "modelpsf": "modelpsf"}
        for requested_type in caltypes:
            kind = requested_type.lower()
            table_type = aliases.get(kind, kind)
            table = self.caldict.get(table_type)
            if table is None:
                continue
            for row in table:
                if np.any((values >= row["mjd1"]) & (values <= row["mjd2"])):
                    root = self.node(kind, row["name"])
                    if root is not None:
                        roots.add(root)
        return roots

    def resolve(
        self, roots: Iterable[CalibrationNode | tuple[str, object]]
    ) -> CalibrationGraph:
        """Recursively resolve all prerequisites, checking for cycles."""

        normalized = set()
        for item in roots:
            if isinstance(item, CalibrationNode):
                normalized.add(CalibrationNode(item.caltype, _canonical_name(item.name)))
            else:
                node = self.node(str(item[0]), item[1])
                if node is not None:
                    normalized.add(node)
        graph = CalibrationGraph(roots=normalized)
        visiting: set[CalibrationNode] = set()

        def visit(node: CalibrationNode) -> None:
            if node in visiting:
                raise ValueError(f"Calibration dependency cycle reaches {node}")
            if node in graph.dependencies:
                return
            visiting.add(node)
            # An existing product is a satisfied leaf.  Its own prerequisites
            # are not execution requirements and must not trigger rebuilding
            # older calibrations outside the requested MJD range.
            if self.exists is not None and self.exists(node):
                graph.existing.add(node)
                graph.dependencies[node] = set()
                visiting.remove(node)
                return
            dependencies = self.direct_dependencies(node)
            graph.dependencies[node] = dependencies
            for dependency in dependencies:
                visit(dependency)
            visiting.remove(node)

        for root in sorted(normalized):
            visit(root)
        return graph
