"""Immutable settings for reconstruction, diagnostics and regularisation."""

from dataclasses import dataclass, fields
from numbers import Real
from typing import ClassVar, Literal
import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _real(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return float(value)


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True)
class FourierOptions:
    """Validated Fourier inversion settings, independent of CUDA resources.

    Defaults use quantitative normalization and centered odd-width cropping.
    legacy_scaling=True restores the previous gain and crop. center_size=0 selects the
    original gather kernel. Chunk counts are upper bounds; minimum-memory
    modes process one detector row / packed slice at a time.
    """

    filter_type: str = "shepp"
    cutoff_freq: float = 1.0
    center_size: int = 32768
    block_dim: tuple[int, int] = (16, 16)
    block_dim_center: tuple[int, int] = (32, 4)
    chunk_count: int = 4
    filter_vol_chunk_count: int = 4
    filter_proj_chunk_count: int = 4
    oversampling_level: int = 4
    power_of_2_oversampling: bool = True
    power_of_2_cropping: bool = False
    min_mem_usage_filter: bool = True
    min_mem_usage_ifft2: bool = True
    padding: int = 0
    recon_mask_radius: float | None = None
    legacy_scaling: bool = False

    def __post_init__(self):
        if self.filter_type not in (
            "none",
            "ramp",
            "shepp",
            "cosine",
            "cosine2",
            "hamming",
            "hann",
            "parzen",
        ):
            raise ValueError("unknown Fourier filter_type")
        _real(self.cutoff_freq, "cutoff_freq")
        for name in ("center_size", "padding"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be a nonnegative integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "chunk_count",
            "filter_vol_chunk_count",
            "filter_proj_chunk_count",
            "oversampling_level",
        ):
            _positive_integer(getattr(self, name), name)
        if self.oversampling_level < 3:
            raise ValueError("oversampling_level must be at least 3")
        for name in (
            "legacy_scaling",
            "power_of_2_oversampling",
            "power_of_2_cropping",
            "min_mem_usage_filter",
            "min_mem_usage_ifft2",
        ):
            _boolean(getattr(self, name), name)
        for name in ("block_dim", "block_dim_center"):
            block = tuple(getattr(self, name))
            if len(block) != 2:
                raise ValueError(f"{name} must contain two dimensions")
            block = tuple(_positive_integer(value, name) for value in block)
            if block[0] * block[1] > 1024:
                raise ValueError(f"{name} must use at most 1024 threads")
            object.__setattr__(self, name, block)
        if self.recon_mask_radius is not None:
            _real(self.recon_mask_radius, "recon_mask_radius")


@dataclass(frozen=True)
class Diagnostics:
    """Sampling intervals in completed outer iterations, across run() calls.

    Energy requires return_energy=True; the initial image and each geometry
    restart are always sampled. residual_interval controls ADMM history and
    verbose reporting; None disables both. convergence_interval controls ADMM
    stopping checks when tolerances are supplied, independently of reporting.
    Defaults preserve sampling/checking every iteration.
    """

    energy_interval: int = 1
    residual_interval: int | None = 1
    convergence_interval: int = 1

    def __post_init__(self):
        _positive_integer(self.energy_interval, "energy_interval")
        if self.residual_interval is not None:
            _positive_integer(self.residual_interval, "residual_interval")
        _positive_integer(self.convergence_interval, "convergence_interval")


@dataclass(frozen=True)
class DataFidelity:
    """Data term for FISTA/ADMM. PWLS and SWLS require ``weights``.

    Weights approximate inverse variances of the supplied line integrals;
    photon counts are a common approximation. They must be finite, nonnegative
    and include a positive value. Zero weights exclude measurements. Weights
    are divided by their maximum without changing the supplied array, so the
    regularisation strength applies to the normalised data objective.
    ``beta_swls`` controls stripe suppression for SWLS only.
    """

    method: Literal["LS", "PWLS", "SWLS"] = "LS"
    beta_swls: float = 0.1

    def __post_init__(self):
        if self.method not in ("LS", "PWLS", "SWLS"):
            raise ValueError("fidelity method must be LS, PWLS or SWLS")
        _real(self.beta_swls, "beta_swls", allow_zero=True)


@dataclass(frozen=True)
class _RegulariserSettings:
    """Validated execution settings, built only by Regularisation._settings.

    Keep solver scaling on the public Regularisation object. Resolve these
    settings afterwards so the execution layer sees the scaled strength and
    only parameters used by its selected method.
    """

    strength: float
    iterations: int
    half_precision: bool


@dataclass(frozen=True)
class _ROFSettings(_RegulariserSettings):
    method: ClassVar[Literal["ROF_TV"]] = "ROF_TV"
    time_step: float


@dataclass(frozen=True)
class _PDSettings(_RegulariserSettings):
    method: ClassVar[Literal["PD_TV"]] = "PD_TV"
    tv_type: Literal[0, 1]
    lipschitz_constant: float


@dataclass(frozen=True)
class _SBSettings(_RegulariserSettings):
    method: ClassVar[Literal["SB_TV"]] = "SB_TV"
    tv_type: Literal[0, 1]


@dataclass(frozen=True)
class _FGPSettings(_RegulariserSettings):
    method: ClassVar[Literal["FGP_TV"]] = "FGP_TV"
    tv_type: Literal[0, 1]


@dataclass(frozen=True)
class _TGVSettings(_RegulariserSettings):
    method: ClassVar[Literal["TGV"]] = "TGV"
    alpha1: float
    alpha0: float
    lipschitz_constant: float


@dataclass(frozen=True)
class _NDFSettings(_RegulariserSettings):
    method: ClassVar[Literal["NDF"]] = "NDF"
    time_step: float
    sigma: float
    penalty_type: Literal[1, 2, 3, 4, 5]


_SETTINGS_TYPES = {
    settings.method: settings
    for settings in (
        _ROFSettings,
        _PDSettings,
        _SBSettings,
        _FGPSettings,
        _TGVSettings,
        _NDFSettings,
    )
}


@dataclass(frozen=True)
class Regularisation:
    """Regulariser settings; pass ``None`` to a solver to disable regularisation.

    ``strength`` is the regularisation parameter. ``time_step`` applies to
    ROF_TV, while ``lipschitz_constant`` applies to PD_TV.
    ``tv_type`` applies to PD_TV, SB_TV and FGP_TV.
    ``tv_type=0`` selects isotropic TV and 1 selects anisotropic TV.
    ``alpha1`` and ``alpha0`` weight TGV first/second-order terms. TGV is
    isotropic only; its step bound uses ``lipschitz_constant`` as a lower bound.
    NDF uses ``time_step``, ``sigma`` (zero selects linear diffusion) and
    ``penalty_type`` (1--5); it supports float32 only.
    The regulariser always runs on the reconstructor's selected device.

    This constructor retains its shared fields for compatibility. All supplied
    values are validated, including fields unused by the selected method.
    Internally, execution and energy evaluation use method-specific settings
    containing only the applicable fields.
    """

    method: Literal["ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV", "NDF"] = "PD_TV"
    strength: float = 0.001
    iterations: int = 150
    time_step: float = 0.005
    lipschitz_constant: float = 12.0
    tv_type: Literal[0, 1] = 0
    half_precision: bool = False
    alpha1: float = 1.0
    alpha0: float = 2.0
    sigma: float = 0.01
    penalty_type: Literal[1, 2, 3, 4, 5] = 1

    def __post_init__(self):
        if self.method not in ("ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV", "NDF"):
            raise ValueError(
                "regularisation method must be ROF_TV, PD_TV, SB_TV, FGP_TV, TGV or NDF"
            )
        _real(self.strength, "strength", allow_zero=True)
        if self.strength > float(np.finfo(np.float32).max):
            raise ValueError("regularisation strength must fit in float32")
        _positive_integer(self.iterations, "regularisation iterations")
        _real(self.time_step, "time_step")
        _real(self.lipschitz_constant, "regularisation lipschitz_constant")
        if (
            isinstance(self.tv_type, (bool, np.bool_))
            or not isinstance(self.tv_type, (int, np.integer))
            or self.tv_type not in (0, 1)
        ):
            raise ValueError("tv_type must be 0 or 1")
        _boolean(self.half_precision, "half_precision")
        _real(self.alpha1, "alpha1")
        _real(self.alpha0, "alpha0")
        _real(self.sigma, "sigma", allow_zero=True)
        if (
            isinstance(self.penalty_type, (bool, np.bool_))
            or not isinstance(self.penalty_type, (int, np.integer))
            or self.penalty_type not in (1, 2, 3, 4, 5)
        ):
            raise ValueError("penalty_type must be an integer from 1 to 5")
        if self.method == "NDF" and self.half_precision:
            raise ValueError("NDF supports float32 only; set half_precision=False")
        if self.method == "TGV" and (
            max(self.alpha1, self.alpha0) > float(np.finfo(np.float32).max)
            or min(self.alpha1, self.alpha0) < float(np.finfo(np.float32).tiny)
        ):
            raise ValueError("TGV alpha weights must fit in normal float32 range")
        if self.method == "TGV" and self.tv_type != 0:
            raise ValueError("TGV supports isotropic tv_type=0 only")

    @property
    def _settings(
        self,
    ) -> (
        _ROFSettings
        | _PDSettings
        | _SBSettings
        | _FGPSettings
        | _TGVSettings
        | _NDFSettings
    ):
        """Resolve validated public options into the selected method's settings.

        Resolve on use rather than caching: dataclasses.replace() is used by
        solvers to scale strength without changing the original objective.
        """
        settings_type = _SETTINGS_TYPES[self.method]
        return settings_type(
            **{field.name: getattr(self, field.name) for field in fields(settings_type)}
        )


def _standalone_regularisation(
    method, strength, iterations, half_precision, nonneg, **parameters
):
    """Use the public settings contract for standalone regulariser calls.

    Legacy ``nonneg`` arguments accept integer 0/1 and Python booleans.
    ROF's boolean-only ``nonnegativity`` is checked by its public wrapper.
    """
    from numbers import Integral

    if not isinstance(nonneg, (Integral, np.bool_)) or nonneg not in (0, 1):
        raise ValueError("nonneg must be 0 or 1")
    return Regularisation(
        method=method,
        strength=strength,
        iterations=iterations,
        half_precision=half_precision,
        **parameters,
    )
