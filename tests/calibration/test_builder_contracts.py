"""Static contracts shared by the independently runnable builders."""

import ast
from pathlib import Path

import pytest


CAL_DIR = (
    Path(__file__).parents[2]
    / "python" / "apogee_drp" / "apred" / "cal"
)
BUILDERS = sorted(
    path for path in CAL_DIR.glob("mk*.py") if path.name != "__init__.py"
)


@pytest.mark.parametrize("path", BUILDERS, ids=lambda path: path.stem)
def test_builder_module_parses(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", BUILDERS, ids=lambda path: path.stem)
def test_builder_module_defines_matching_function(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert path.stem in functions


@pytest.mark.parametrize("path", BUILDERS, ids=lambda path: path.stem)
def test_builder_has_docstring(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == path.stem
    )
    assert ast.get_docstring(function), f"{path.stem} needs a docstring"


@pytest.mark.parametrize("path", BUILDERS, ids=lambda path: path.stem)
def test_builder_has_clobber_control(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == path.stem
    )
    arguments = {arg.arg for arg in function.args.args}
    assert "clobber" in arguments, f"{path.stem} lacks clobber"

