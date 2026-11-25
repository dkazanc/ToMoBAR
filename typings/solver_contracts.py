"""Static conformance checks for production solver services.

Pyright checks these assignments without importing CUDA at runtime.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tomobar._solvers.interface import SolverOperator, SolverReconstructor
    from tomobar.projectors.operator import ProjectionOperator
    from tomobar.reconstruction.iterative import IterativeReconstructor

    def check_operator(operator: ProjectionOperator) -> SolverOperator:
        """Keep the concrete operator compatible with solver calls."""
        return operator

    def check_reconstructor(rec: IterativeReconstructor) -> SolverReconstructor:
        """Keep the public reconstructor compatible with the solver engine."""
        return rec
