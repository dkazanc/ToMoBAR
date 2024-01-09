import os
from typing import List, Tuple

try:
    import cupy as cp
except ImportError:
    print(
        "Cupy library is a required dependency for this part of the code, please install"
    )


def load_cuda_module(
    file: str, name_expressions: List[str] = None, options: Tuple[str] = tuple()
) -> cp.RawModule:
    """Load a CUDA module file, i.e. a .cu file, from the file system,
    compile it, and return is as a CuPy RawModule for further
    processing.
    """

    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, file + ".cu")
    # insert a preprocessor line directive to assist compiler errors (so line numbers show correctly in output)
    escaped = file.replace("\\", "\\\\")
    code = '#line 1 "{}"\n'.format(escaped)
    with open(file, "r") as f:
        code += f.read()

    return cp.RawModule(
        options=("-std=c++11", *options), code=code, name_expressions=name_expressions
    )
