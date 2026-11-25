"""Check demo keyword names against source APIs without importing CUDA packages.

This is a signature check, not execution of optional datasets or TomoPhantom.
Forwarded Fourier kwargs are checked against FourierOptions, so a typo cannot
hide behind FOURIER_INV's **kwargs parameter.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def api_keywords():
    registry = {}
    paths = (
        "reconstruction/direct.py",
        "reconstruction/iterative.py",
        "reconstruction/iterative_methods.py",
        "reconstruction/iterative_state.py",
        "projectors/operator.py",
        "geometry.py",
        "reconstruction/base.py",
        "options.py",
        "regularisers.py",
        "preprocessing.py",
        "postprocessing.py",
        "visualisation.py",
    )
    for relative in paths:
        tree = ast.parse((ROOT / "tomobar" / relative).read_text())
        for item in tree.body:
            if isinstance(item, ast.FunctionDef):
                registry[item.name] = item
            elif isinstance(item, ast.ClassDef):
                methods = {
                    n.name: n for n in item.body if isinstance(n, ast.FunctionDef)
                }
                registry[item.name] = methods.get("__init__")
                registry.update(
                    {item.name + "." + name: method for name, method in methods.items()}
                )
                if item.name in (
                    "FourierOptions",
                    "DataFidelity",
                    "Regularisation",
                    "Diagnostics",
                ):
                    registry[item.name] = {
                        n.target.id
                        for n in item.body
                        if isinstance(n, ast.AnnAssign)
                        and isinstance(n.target, ast.Name)
                    }
    for cls in ("DirectReconstructor", "IterativeReconstructor"):
        for name, body in list(registry.items()):
            if name.startswith("GeometryReconstruction."):
                registry.setdefault(cls + name[len("GeometryReconstruction") :], body)
    # IterativeReconstructor exposes these methods through private implementation mixins.
    for mixin in ("_IterativeMethods", "_IterativeStateMethods"):
        for name, body in list(registry.items()):
            if name.startswith(mixin + "."):
                registry.setdefault(
                    "IterativeReconstructor." + name[len(mixin) + 1 :], body
                )
    allowed = {}
    for name, body in registry.items():
        if body is None:
            continue
        if isinstance(body, set):
            allowed[name] = body
            continue
        keys = {
            arg.arg
            for arg in body.args.posonlyargs + body.args.args + body.args.kwonlyargs
        }
        if body.args.kwarg:
            kwargs_name = body.args.kwarg.arg
            keys |= {
                node.args[0].value
                for node in ast.walk(body)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == kwargs_name
                and node.func.attr in ("get", "pop")
                and node.args
                and isinstance(node.args[0], ast.Constant)
            }
            if name.endswith(".FOURIER_INV"):
                keys |= registry["FourierOptions"]
            if name.endswith(".FBP"):
                keys.add("recon_mask_radius")  # Applied by postprocess_reconstruction.
        allowed[name] = keys
    # initialise_solver is defined before solver methods in the source.
    allowed["IterativeReconstructor.initialise_solver"] |= set().union(
        *(
            allowed["IterativeReconstructor." + method]
            for method in ("Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM")
        )
    )
    allowed["IterativeReconstructor.initialise_solver"] -= {
        "iterations",
        "recon_mask_radius",
    }
    return allowed


API_KEYWORDS = api_keywords()


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "demos").rglob("*.py")),
    ids=lambda p: str(p.relative_to(ROOT / "demos")),
)
def test_demo_keywords_match_current_api(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    aliases, receivers = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "tomobar."
        ):
            aliases.update({item.asname or item.name: item.name for item in node.names})
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.withitem)
            and isinstance(node.context_expr, ast.Call)
            and isinstance(node.context_expr.func, ast.Name)
            and isinstance(node.optional_vars, ast.Name)
        ):
            receivers[node.optional_vars.id] = aliases.get(node.context_expr.func.id)
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    receivers[target.id] = aliases.get(node.value.func.id)
    issues = []

    def check(name, keywords, lineno):
        if name in API_KEYWORDS:
            unknown = {kw.arg for kw in keywords if kw.arg is not None} - API_KEYWORDS[
                name
            ]
            if unknown:
                issues.append(
                    f"line {lineno}: {name}: unknown arguments {sorted(unknown)}"
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = aliases.get(node.func.id)
            elif isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                cls = receivers.get(node.func.value.id) or aliases.get(
                    node.func.value.id
                )
                name = f"{cls}.{node.func.attr}"
            check(name, node.keywords, node.lineno)
        # The peppers demo passes settings dictionaries through a denoiser loop.
        if isinstance(node, ast.Tuple) and len(node.elts) == 3:
            function, settings = node.elts[1:]
            if (
                isinstance(function, ast.Name)
                and isinstance(settings, ast.Call)
                and isinstance(settings.func, ast.Name)
                and settings.func.id == "dict"
            ):
                check(aliases.get(function.id), settings.keywords, node.lineno)
    assert not issues, f"{path.name}:\n" + "\n".join(issues)
