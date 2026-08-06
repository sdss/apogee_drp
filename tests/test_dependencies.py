import numpy as np

from apogee_drp.apred.cal.dependencies_v3 import (
    CalibrationDependencyResolver,
    CalibrationNode,
    calibration_dependency_tree,
)


def table(dtype, rows):
    return np.array(rows, dtype=np.dtype(dtype))


class FakeLoad:
    def cmjd(self, exposure):
        return int(exposure) // 1000


def sample_caldict():
    common = [("mjd1", int), ("mjd2", int), ("name", "U20")]
    return {
        "det": table(common + [("linid", int)], [(10, 99, "10001", 10001)]),
        "dark": table(common + [("frames", "U40")], [(10, 99, "12001", "12001-12003")]),
        "flat": table(common + [("frames", "U40"), ("nrep", int), ("dithered", int)],
                      [(10, 99, "13001", "13001-13003", 1, 0)]),
        "bpm": table(common + [("darkid", int), ("flatid", int)],
                     [(10, 99, "14001", 12001, 13001)]),
        "fiber": table(common, [(10, 99, "15001")]),
        "sparse": table(common + [("frames", "U40"), ("darkframes", "U40"),
                                    ("dmax", int), ("maxread", "U20")],
                        [(10, 99, "16001", "16001-16003", "0", 7, "30,30,30")]),
        "littrow": table(common + [("psfid", int)], [(10, 99, "17001", 17001)]),
        "modelpsf": table(common + [("sparse", "U20"), ("psf", "U20")],
                          [(10, 99, "18001", "16001", "18002")]),
    }


def test_sparse_dependency_closure_has_no_fiber_sparse_cycle():
    resolver = CalibrationDependencyResolver(sample_caldict(), FakeLoad())
    graph = resolver.resolve([("sparse", 16001)])

    sparse = CalibrationNode("sparse", "16001")
    fiber = CalibrationNode("fiber", "15001")
    assert fiber in graph.dependencies[sparse]
    assert sparse not in graph.dependencies[fiber]
    assert CalibrationNode("detector", "10001") in graph.nodes


def test_modelpsf_tree_includes_old_sparse_and_psf_dependencies():
    resolver = CalibrationDependencyResolver(sample_caldict(), FakeLoad())
    graph = resolver.resolve([("modelpsf", "18001")])

    assert CalibrationNode("sparse", "16001") in graph.nodes
    assert CalibrationNode("psf", "18002") in graph.nodes
    levels = graph.topological_levels()
    positions = {node: level for level, nodes in enumerate(levels) for node in nodes}
    assert positions[CalibrationNode("fiber", "15001")] < positions[CalibrationNode("sparse", "16001")]


def test_existing_products_are_removed_only_from_missing_plan():
    existing = {CalibrationNode("dark", "12001")}
    resolver = CalibrationDependencyResolver(
        sample_caldict(), FakeLoad(), exists=lambda node: node in existing
    )
    graph = resolver.resolve([("bpm", "14001")])

    assert CalibrationNode("dark", "12001") in graph.nodes
    assert CalibrationNode("dark", "12001") not in graph.missing
    assert CalibrationNode("flat", "13001") in graph.missing


def test_existing_product_prunes_its_dependencies():
    existing = {CalibrationNode("bpm", "14001")}
    resolver = CalibrationDependencyResolver(
        sample_caldict(), FakeLoad(), exists=lambda node: node in existing
    )
    graph = resolver.resolve([("bpm", "14001")])

    assert graph.nodes == {CalibrationNode("bpm", "14001")}
    assert graph.missing == set()


def test_required_names_groups_missing_jobs_for_mastercal_stages():
    resolver = CalibrationDependencyResolver(sample_caldict(), FakeLoad())
    graph = resolver.resolve([("sparse", "16001")])

    grouped = graph.required_by_type()
    assert grouped["sparse"] == ["16001"]
    assert grouped["fiber"] == ["15001"]
    assert graph.required_names("dark") == {"12001"}
    assert graph.is_required("dark", "00012001")


def test_roots_for_mjds_selects_only_overlapping_products():
    resolver = CalibrationDependencyResolver(sample_caldict(), FakeLoad())
    roots = resolver.roots_for_mjds([12], ["detector", "dark", "fiber"])

    assert roots == {
        CalibrationNode("detector", "10001"),
        CalibrationNode("dark", "12001"),
        CalibrationNode("fiber", "15001"),
    }


def test_public_function_plans_range(monkeypatch, tmp_path):
    from apogee_drp.apred import mkcal

    monkeypatch.setattr(mkcal, "readcal", lambda filename: sample_caldict())
    graph = calibration_dependency_tree(
        12,
        13,
        load=FakeLoad(),
        caltypes=["sparse"],
        calfile=str(tmp_path / "unused.par"),
        check_exists=False,
        print_tree=False,
    )

    assert graph.roots == {CalibrationNode("sparse", "16001")}
    assert CalibrationNode("fiber", "15001") in graph.nodes
    assert CalibrationNode("detector", "10001") in graph.nodes
