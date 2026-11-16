"""Assemble production wrapper/algorithm ASTs for CPU CUDA-kernel adapters.

Only dependency injection changes: tests supply NumPy storage and compiled CPU
launchers. Algorithm bodies and shared preparation come from production files.
"""

import ast
import copy
from pathlib import Path


def regulariser_tree():
    root = Path(__file__).resolve().parents[2] / "tomobar"
    facade = ast.parse((root / "regularisers.py").read_text())
    common = ast.parse((root / "_regularisers/common.py").read_text())
    helpers = [n for n in common.body if isinstance(n, ast.FunctionDef)]
    for helper in helpers:
        helper.returns = None
        for arg in helper.args.args:
            arg.annotation = None
    for function in facade.body:
        if not isinstance(function, ast.FunctionDef) or not function.name.endswith(
            "_cupy"
        ):
            continue
        calls = [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "run"
        ]
        if not calls:
            continue
        stem = calls[0].func.value.id.removeprefix("_")
        algorithm = ast.parse((root / "_regularisers" / (stem + ".py")).read_text())
        run = next(n for n in algorithm.body if isinstance(n, ast.FunctionDef))
        run.name = "_run_algorithm"
        calls[0].func = ast.Name(id=run.name, ctx=ast.Load())
        function.body[1:1] = [*copy.deepcopy(helpers), run]
    proximal = ast.parse((root / "_regularisers/proximal.py").read_text())
    dispatch = next(n for n in proximal.body if isinstance(n, ast.FunctionDef))

    class InjectRegularisers(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "regularisers":
                return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
            return self.generic_visit(node)

    facade.body += [InjectRegularisers().visit(dispatch), *helpers]
    return ast.fix_missing_locations(facade)
