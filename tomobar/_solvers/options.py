"""Immutable, method-specific solver configuration without CUDA dependencies."""

from dataclasses import dataclass
from typing import Literal

from tomobar.options import DataFidelity, Diagnostics, Regularisation
from tomobar._validation import _boolean, _positive_integer, _real


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Settings used by every solver; subclasses add only applicable fields."""

    nonnegativity: bool = False
    diagnostics: Diagnostics = Diagnostics()

    def __post_init__(self):
        _boolean(self.nonnegativity, "nonnegativity")
        if not isinstance(self.diagnostics, Diagnostics):
            raise TypeError("diagnostics must be a Diagnostics instance")


@dataclass(frozen=True, slots=True)
class LandweberOptions(SolverOptions):
    """Validated gradient-descent step size and constraint settings."""

    step_size: float = 1e-5

    def __post_init__(self):
        SolverOptions.__post_init__(self)
        object.__setattr__(self, "step_size", _real(self.step_size, "step_size"))


@dataclass(frozen=True, slots=True)
class SIRTOptions(SolverOptions):
    """Validated simultaneous-iteration constraint settings."""

    pass


@dataclass(frozen=True, slots=True)
class CGLSOptions(SolverOptions):
    """Validated settings for conjugate-gradient least squares."""

    pass


@dataclass(frozen=True, slots=True)
class _RegularisedOptions(SolverOptions):
    regularisation: Regularisation | None = None

    def __post_init__(self):
        SolverOptions.__post_init__(self)
        if self.regularisation is not None and not isinstance(
            self.regularisation, Regularisation
        ):
            raise TypeError("regularisation must be a Regularisation instance or None")


@dataclass(frozen=True, slots=True)
class _QuadraticOptions(_RegularisedOptions):
    fidelity: DataFidelity = DataFidelity()
    lipschitz_constant: float | None = None

    def __post_init__(self):
        _RegularisedOptions.__post_init__(self)
        if not isinstance(self.fidelity, DataFidelity):
            raise TypeError("fidelity must be a DataFidelity instance")
        if self.lipschitz_constant is not None:
            object.__setattr__(
                self,
                "lipschitz_constant",
                _real(self.lipschitz_constant, "lipschitz_constant"),
            )


@dataclass(frozen=True, slots=True)
class OSEMOptions(_RegularisedOptions):
    """Validated settings for ordered-subset expectation maximisation."""

    pass


@dataclass(frozen=True, slots=True)
class FISTAOptions(_QuadraticOptions):
    """Validated settings for accelerated proximal-gradient iterations."""

    pass


@dataclass(frozen=True, slots=True)
class ADMMOptions(_QuadraticOptions):
    """Validated ADMM splitting, data-solve and stopping settings."""

    rho: float = 1.0
    data_solver: Literal["linearised", "cg"] = "linearised"
    cg_iterations: int = 5
    cg_tolerance: float = 1e-4
    relaxation: float = 1.6
    verbose: bool = False
    absolute_tolerance: float | None = None
    relative_tolerance: float | None = None

    def __post_init__(self):
        _QuadraticOptions.__post_init__(self)
        object.__setattr__(self, "rho", _real(self.rho, "rho"))
        object.__setattr__(self, "relaxation", _real(self.relaxation, "relaxation"))
        if self.relaxation >= 2:
            raise ValueError("relaxation must be between 0 and 2")
        _boolean(self.verbose, "verbose")
        if self.data_solver not in ("linearised", "cg"):
            raise ValueError("data_solver must be linearised or cg")
        object.__setattr__(
            self,
            "cg_iterations",
            _positive_integer(self.cg_iterations, "cg_iterations"),
        )
        object.__setattr__(
            self,
            "cg_tolerance",
            _real(self.cg_tolerance, "cg_tolerance", allow_zero=True),
        )
        if self.cg_tolerance >= 1:
            raise ValueError("cg_tolerance must be less than one")
        if (self.absolute_tolerance is None) != (self.relative_tolerance is None):
            raise ValueError("provide both absolute_tolerance and relative_tolerance")
        if self.absolute_tolerance is not None:
            object.__setattr__(
                self,
                "absolute_tolerance",
                _real(self.absolute_tolerance, "absolute_tolerance", allow_zero=True),
            )
            object.__setattr__(
                self,
                "relative_tolerance",
                _real(self.relative_tolerance, "relative_tolerance", allow_zero=True),
            )


OPTION_TYPES = {
    "Landweber": LandweberOptions,
    "SIRT": SIRTOptions,
    "CGLS": CGLSOptions,
    "FISTA": FISTAOptions,
    "ADMM": ADMMOptions,
    "OSEM": OSEMOptions,
}


def solver_options(method, **options) -> SolverOptions:
    """Reject unsupported fields, including explicitly supplied default values."""
    if method not in OPTION_TYPES:
        raise ValueError(f"method must be one of {tuple(OPTION_TYPES)}")
    return OPTION_TYPES[method](**options)
