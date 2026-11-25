"""Immutable, validated options shared by reconstruction methods.

Use :class:`DataFidelity` for the FISTA/ADMM data term,
:class:`Regularisation` for the image penalty, :class:`Diagnostics` for sampling
intervals, and :class:`FourierOptions` for direct Fourier inversion. These
objects configure different parts of reconstruction; similarly named fields
such as regulariser and solver Lipschitz constants are not interchangeable.

Constructing options requires neither CUDA nor a projector. Instances are frozen;
use ``dataclasses.replace(options, field=value)`` to make a modified copy.
"""

from dataclasses import dataclass, fields
from typing import ClassVar, Literal
import numpy as np

from tomobar._validation import _positive_integer, _real, _boolean


@dataclass(frozen=True)
class FourierOptions:
    """Settings for :meth:`~tomobar.reconstruction.direct.DirectReconstructor.FOURIER_INV`.

    These options apply to direct Fourier inversion, not FBP or the iterative
    ``backend="usfft"`` projector. Defaults are suitable for a first run;
    chunk and kernel settings primarily control execution and memory use.

    Args:
        filter_type (str): Frequency filter: ``"none"``, ``"ramp"``,
            ``"shepp"`` (default), ``"cosine"``, ``"cosine2"``, ``"hamming"``,
            ``"hann"`` or ``"parzen"``. Windowed filters modify the ramp
            response; ``"none"`` omits it.
        cutoff_freq (float): Positive filter-response multiplier, default 1.0.
            In this Fourier implementation it scales the response amplitude;
            it is not the FBP SINC shape parameter or a Nyquist cutoff fraction.
        center_size (int): Nonnegative upper bound on the central interpolation-grid
            width, default 32768. Capped at twice the working detector width.
            Effective sizes below 192, including zero, select the gather-only
            path. Larger central regions require more lookup storage.
        block_dim (tuple[int, int]): CUDA block dimensions for the gather path,
            default (16, 16). Both dimensions must be positive, with at most
            1024 threads per block. Tune only when profiling kernel execution.
        block_dim_center (tuple[int, int]): CUDA block dimensions for the
            central interpolation path, default (32, 4). Same constraints as block_dim;
            unused when the central interpolation path is disabled.
        chunk_count (int): Positive upper bound on inverse-FFT slice chunks,
            default 4. Used only when min_mem_usage_ifft2=False. More chunks
            generally reduce temporary memory at the cost of smaller batches.
        filter_vol_chunk_count (int): Positive upper bound on detector-row
            chunks during filtering, default 4. Used only when
            min_mem_usage_filter=False.
        filter_proj_chunk_count (int): Positive upper bound on view chunks
            during filtering, default 4. Applies in either filter-memory mode.
        oversampling_level (int): Integer FFT-width multiplier of at least 3,
            default 4. Used only when power_of_2_oversampling=False; the FFT
            width is also large enough for the requested detector padding.
        power_of_2_oversampling (bool): Default True chooses a power-of-two
            filtering FFT width covering at least three times the even detector
            width and any requested padding, instead of oversampling_level.
        power_of_2_cropping (bool): Default False. True rounds the working
            detector width up to the next power of two when it is already
            within 10 percent of that size. The final volume shape is unchanged.
        min_mem_usage_filter (bool): Default True filters one detector row at
            a time, ignoring filter_vol_chunk_count. False uses row batches.
        min_mem_usage_ifft2 (bool): Default True transforms one packed slice
            at a time, ignoring chunk_count. Each packed slice holds two real
            detector rows. False uses larger inverse-FFT batches.
        padding (int): Additional nonnegative pixels per detector side,
            default 0. Added to the reconstructor's detector_padding for this
            Fourier calculation; it does not change the acquired input shape.
        recon_mask_radius (float or None): Positive output circular-mask
            parameter, default None for no mask. A value of 1.0 selects the
            inscribed circle; smaller values mask more of each slice.
        legacy_scaling (bool): Default False uses quantitative normalisation
            and centred odd-width cropping. True restores the former amplitude
            (8/pi times the quantitative scale) and odd-width crop convention.

    Notes:
        Chunk counts are upper bounds: actual batch counts depend on array
        dimensions. Unused options are still validated. Use
        :meth:`~tomobar.reconstruction.direct.DirectReconstructor.plan_fourier`
        to inspect dimensions and workspace sizes without allocating GPU arrays.
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

    def __post_init__(self) -> None:
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
    """Sampling schedules for iterative reconstruction diagnostics.

    All intervals count completed outer iterations, including all subsets, and
    continue across successive ``run()`` calls. They do not change solver step
    sizes or the number of iterations requested.

    Args:
        energy_interval (int): Positive interval between energy evaluations,
            default 1. Applies to all iterative methods when return_energy=True;
            otherwise it has no effect. The initial image and each geometry
            restart are always sampled. Larger intervals reduce the extra
            forward-projection work used for energy monitoring.
        residual_interval (int or None): ADMM only. Positive interval between
            residual-history samples and verbose reports, default 1. None
            disables both, without disabling convergence checks.
        convergence_interval (int): ADMM only. Positive interval between
            stopping checks when absolute_tolerance and relative_tolerance are
            supplied, default 1. Larger values can delay detecting convergence.
            It is independent of residual_interval and has no stopping effect
            when tolerances are omitted.

    Examples:
        Record energy every ten iterations while checking ADMM convergence
        every iteration, without retaining residual history::

            diagnostics = Diagnostics(energy_interval=10, residual_interval=None)
    """

    energy_interval: int = 1
    residual_interval: int | None = 1
    convergence_interval: int = 1

    def __post_init__(self) -> None:
        _positive_integer(self.energy_interval, "energy_interval")
        if self.residual_interval is not None:
            _positive_integer(self.residual_interval, "residual_interval")
        _positive_integer(self.convergence_interval, "convergence_interval")


@dataclass(frozen=True)
class DataFidelity:
    """Data-term settings for FISTA and ADMM.

    These settings do not select the data model for Landweber, SIRT, CGLS or
    OSEM. Measurement weights are passed to the solver as ``weights=...``,
    rather than stored in this object.

    Args:
        method (str): ``"LS"`` (default) uses unweighted least squares.
            ``"PWLS"`` uses weighted least squares. ``"SWLS"`` also models
            stripe-like detector errors shared across views. PWLS and SWLS
            require weights; LS does not accept them.
        beta_swls (float): SWLS only. Nonnegative stripe-variable penalty,
            default 0.1. Larger values penalise the stripe correction more
            strongly, bringing the data term closer to PWLS; smaller values
            allow a larger stripe correction. Zero removes this penalty.
            Validated but unused for LS and PWLS.

    Notes:
        Weights approximate inverse variances of the supplied line integrals;
        photon counts are a common approximation. They must match projection
        shape and axis order, be finite and nonnegative, and contain a positive
        value. Zero weights exclude measurements. The solver divides weights
        by their maximum without changing the supplied array, so regularisation
        strength applies to the normalised data objective.

        See :func:`~tomobar.energy.data_energy` for the objective
        definitions and :doc:`/howto/regularisation` for a reconstruction example.
    """

    method: Literal["LS", "PWLS", "SWLS"] = "LS"
    beta_swls: float = 0.1

    def __post_init__(self) -> None:
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
    """Image-penalty settings for FISTA, ADMM and OSEM.

    Pass this object as ``regularisation=...`` to a solver, or pass None to
    disable regularisation. Only the fields applicable to ``method`` affect
    execution. All supplied fields are validated, including unused ones.

    Args:
        method (str): Regulariser: ``"ROF_TV"`` (Rudin--Osher--Fatemi TV),
            ``"PD_TV"`` (primal-dual TV, default), ``"SB_TV"`` (split-Bregman
            TV), ``"FGP_TV"`` (fast-gradient projection TV), ``"TGV"``
            (second-order total generalised variation), or ``"NDF"``
            (nonlinear diffusion).
        strength (float): Overall nonnegative penalty weight, default 0.001.
            Applies to every method; larger weights favour stronger
            regularisation. Zero skips the penalty update. FISTA and ADMM
            rescale this weight internally for their proximal steps; the
            supplied settings object is unchanged. Must fit in float32.
        iterations (int): Positive number of inner regulariser iterations per
            application, default 150. Applies to every method. This is separate
            from the reconstruction solver's outer iterations: increasing it
            allows more inner updates and adds work per reconstruction step.
            Finite inner iterations do not guarantee convergence.
        time_step (float): ROF_TV and NDF only. Positive explicit time step,
            default 0.005. Smaller steps may need more inner iterations.
            ROF_TV uses the supplied value; NDF caps it internally for stability.
        lipschitz_constant (float): PD_TV and TGV only. Positive squared-operator
            bound used to choose inner primal-dual steps, default 12.0. The
            implementation raises it to at least 4*ndim for PD_TV or 8*ndim+2
            for TGV. This is not the reconstruction solver's Lipschitz constant
            and is not an estimate of the projection operator norm.
        tv_type (int): PD_TV, SB_TV and FGP_TV only: 0 selects isotropic TV
            (default), 1 selects anisotropic TV. ROF_TV is isotropic; TGV also
            supports only isotropic regularisation and requires tv_type=0.
            Unused for NDF.
        half_precision (bool): Default False. True stores selected auxiliary
            or dual arrays in float16 to reduce memory use; reconstructed images
            remain float32. Available for all TV methods and TGV, but not NDF.
            Lower precision can limit the accuracy of the inner solve.
        alpha1 (float): TGV only. Positive weight of the first-order coupling
            term ``||Du - v||``, default 1.0. Here u is the image and v is TGV's
            auxiliary vector field. It controls how closely v follows the
            image gradient. Validated but unused by every other method.
        alpha0 (float): TGV only. Positive weight of the second-order term
            ``||Ev||``, default 2.0, where E is the symmetrised gradient of v.
            It penalises variation in the auxiliary field, allowing smooth
            intensity ramps rather than favouring only piecewise-constant
            regions. Validated but unused by every other method.
        sigma (float): NDF only. Nonnegative edge threshold parameter,
            default 0.01; the effective threshold is sigma/sqrt(2), in image
            intensity-difference units. Zero selects linear diffusion,
            regardless of penalty_type. Positive values control which local
            differences are treated as edges. Unused by TV and TGV methods.
        penalty_type (int): NDF only. Edge-stopping penalty: 1 Huber (default),
            2 Perona--Malik, 3 Tukey biweight, 4 truncated quadratic, or
            5 truncated Huber. Types 2--5 are nonconvex. Validated but unused
            by TV and TGV methods.

    Notes:
        Method-specific fields are summarised below. Every method also uses
        strength and iterations; half_precision is available except for NDF.

        .. list-table:: Parameters used by each regulariser
           :header-rows: 1
           :widths: 20 55 25

           * - Method
             - Additional fields
             - Published background
           * - ROF_TV
             - ``time_step``
             - [ROF1992]_
           * - PD_TV
             - ``tv_type``, ``lipschitz_constant``
             - [CP2011]_
           * - SB_TV
             - ``tv_type``
             - [GO2009]_
           * - FGP_TV
             - ``tv_type``
             - [FGP2009]_
           * - TGV
             - ``alpha1``, ``alpha0``, ``lipschitz_constant``; ``tv_type`` must be 0
             - [TGV2010]_
           * - NDF
             - ``time_step``, ``sigma``, ``penalty_type``; ``half_precision`` must be False
             - Perona--Malik diffusion [PM1990]_

        For TGV the penalty is ``strength * (alpha1*||Du-v|| + alpha0*||Ev||)``,
        minimised over v. The alpha weights balance its first- and second-order
        terms; strength scales both together. TGV weights must fit in the normal
        positive float32 range. TGV currently requires return_energy=False,
        because image-only TGV energy evaluation is not supported.

        The regulariser runs on the reconstructor's selected device. A solver's
        nonnegativity option controls the image constraint separately. See
        :doc:`/reference/solver_conventions` for outer-solver scaling,
        or :mod:`tomobar.regularisers` for standalone denoising functions.

    Examples:
        Configure TGV-specific weights explicitly::

            tgv = Regularisation(
                method="TGV", strength=0.001, alpha1=1.0, alpha0=2.0, iterations=150
            )

        The same alpha fields have no effect on a PD-TV solve. Configure its
        norm and inner step bound instead::

            tv = Regularisation(method="PD_TV", tv_type=0, lipschitz_constant=12.0)
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

    def __post_init__(self) -> None:
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
) -> Regularisation:
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
