"""Reconstruction class for direct reconstruction methods (2D/3D).

* :func:`RecToolsDIR.forward` and :func:`RecToolsDIR.backproject` Forward/Backward 2D/3D projection (ASTRA-Toolbox)
* :func:`RecToolsDIR.FBP` Filtered Back Projection 2D/3D (ASTRA with the custom built filter).
"""

import cupy as cp
import numpy as np

from tomobar.fourier import _filtersinc3D, _filterramlak3D
from tomobar._sinc_filter import SINC_DEFAULT_CUTOFF, validate_sinc_cutoff
from tomobar._ramlak_filter import validate_ramlak_cutoff
from tomobar.supp.axis_order import _transpose_data_axes, _resolve_data_axes
from tomobar.supp.device import _on_device, _clear_gpu_cache
from tomobar.supp.memory_estimator_helpers import DeviceMemStack
from tomobar.supp.suppTools import check_kwargs
from tomobar.astra_wrappers.astra_tools2d import AstraTools2D
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)
from tomobar.reconstruction_geometry import GeometryReconstruction
from tomobar.operators import ProjectionOperator

from tomobar.options import FourierOptions, _boolean
from tomobar.fourier_plan import (
    FourierPlan,
    build_fourier_plan,
    resolve_fourier_options,
    CENTER_SIZE_MIN,
)
from tomobar._fourier_execution import FourierExecutor

_CENTER_SIZE_MIN = CENTER_SIZE_MIN  # Legacy import used by kernel tests.


class RecToolsDIR(GeometryReconstruction):
    """Direct reconstruction from explicit 2D or 3D acquisition geometry.

    Keyword Args:
        projection_geometry (Parallel2DGeometry or Parallel3DGeometry): Required
            scan geometry. Pair 2D and 3D scans with the corresponding volume type.
        volume_geometry (Volume2DGeometry or Volume3DGeometry): Required working
            grid and reconstruction shape, preserved by direct methods.
        detector_padding (int): Nonnegative edge pixels added per detector side,
            default 0. Creates an internal padded scan without modifying the supplied
            geometry. Public projections retain the acquired detector width.
        device_index (int): Nonnegative CUDA device index, default 0. Projector
            handles are created lazily and reused until close().

    Notes:
        NumPy inputs and outputs of forward(), backproject() and FBP() stay on
        the host while projection and filtering execute on the GPU. CuPy arrays
        stay on the selected GPU. FOURIER_INV() always returns a CuPy image;
        shape-only estimation returns a shape tuple instead.
    """

    def __init__(
        self,
        *,
        projection_geometry: Parallel2DGeometry | Parallel3DGeometry,
        volume_geometry: Volume2DGeometry | Volume3DGeometry,
        detector_padding: int = 0,
        device_index: int = 0,
    ):
        self._init_device(device_index)
        self._init_geometry(projection_geometry, volume_geometry, detector_padding)
        wrapper = AstraTools3D if self.geom == "3D" else AstraTools2D
        self.Atools = wrapper(
            projection_geometry=self.padded_projection_geometry,
            volume_geometry=volume_geometry,
            device_index=self.device_index,
        )
        self._init_operator(ProjectionOperator)

    @_on_device(cp)
    def forward(
        self,
        data: np.ndarray | cp.ndarray,
        *,
        output_axes: list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ) -> np.ndarray | cp.ndarray:
        """Forward-project a 2D image or 3D volume.

        Args:
            data (numpy.ndarray or cupy.ndarray): Real object matching
                volume_geometry.shape, in (Y, X) or (Z, Y, X) order.
                Noncontiguous or non-float32 input is packed as float32.

        Keyword Args:
            output_axes (list[str] or tuple[str, ...] or None): Projection axis labels.
                Default None uses (angles, detX) in 2D and (detY, angles, detX) in 3D.
            data_axes_labels_order (list[str] or tuple[str, ...] or None): Legacy alias
                for output_axes, default None. Conflicting aliases raise ValueError.

        Returns:
            numpy.ndarray or cupy.ndarray: Float32 projections matching the input
            array library and requested axis order. Internal detector padding is
            cropped away to the acquired detector shape.
        """
        native_axes = (
            ("angles", "detX") if self.geom == "2D" else ("detY", "angles", "detX")
        )
        axes = _resolve_data_axes(
            output_axes, kwargs.get("data_axes_labels_order"), native_axes
        )
        self._ensure_open()
        projected = self.Atools.forward(data)
        projected = self._crop_projection(projected)
        return _transpose_data_axes(
            projected, source_axes=native_axes, target_axes=axes
        )

    @_on_device(cp)
    def backproject(
        self,
        data: np.ndarray | cp.ndarray,
        *,
        input_axes: list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ) -> np.ndarray | cp.ndarray:
        """Backproject 2D or 3D measured projections.

        Args:
            data (numpy.ndarray or cupy.ndarray): Real projection data matching the
                acquisition geometry. Non-float32 input is packed as float32.

        Keyword Args:
            input_axes (list[str] or tuple[str, ...] or None): Projection axis labels.
                Default None uses (angles, detX) in 2D and (detY, angles, detX) in 3D.
            data_axes_labels_order (list[str] or tuple[str, ...] or None): Legacy alias
                for input_axes, default None. Conflicting aliases raise ValueError.

        Returns:
            numpy.ndarray or cupy.ndarray: Unnormalised float32 backprojection,
            matching the input array library and volume_geometry.shape. Output axes
            are (Y, X) in 2D and (Z, Y, X) in 3D. No filter or output mask is applied.
        """
        data = self._prepare_projection(
            data,
            array_module=cp,
            input_axes=input_axes,
            legacy_axes=kwargs.get("data_axes_labels_order"),
        )
        return self.Atools.backproject(data)

    @_on_device(cp)
    def FBP(
        self,
        data: np.ndarray | cp.ndarray,
        *,
        input_axes: list[str] | tuple[str, ...] | None = None,
        filter_type: str | None = None,
        cutoff_freq: float | None = None,
        clear_gpu_cache: bool = False,
        legacy_scaling: bool = False,
        **kwargs,
    ) -> np.ndarray | cp.ndarray:
        """Reconstruct 2D or 3D projection data by filtered backprojection.

        Args:
            data (numpy.ndarray or cupy.ndarray): Projection data, converted to
                contiguous float32 storage for reconstruction. Default axis order
                is (angles, detX) in 2D and (detY, angles, detX) in 3D.

        Keyword Args:
            input_axes (list[str] or tuple[str, ...] or None): Axis labels for
                ``data``. Default None uses the order above. Labels must be a
                permutation of the default labels.
            filter_type (str or None): Filter selection. Default None selects
                "SINC" in 3D and ASTRA's "ram-lak" in 2D. In 3D, "RAM-LAK"
                selects the discrete ramp; "ram-lak" and "ramp" are aliases.
                In 2D, ASTRA FBP_CUDA filter names are accepted.
            cutoff_freq (float or None): Default None selects 0.35 for 3D SINC
                and 1.0 for 3D RAM-LAK. For SINC, this is a shape parameter in
                the positive normal float32 range, rather than a fractional
                Nyquist threshold. Lower values give sharper reconstructions;
                values above one are supported. For RAM-LAK, this is a fraction
                of detector Nyquist in (0, 1]. Unused in 2D; use ``filter_d``
                for ASTRA filters that support a frequency cutoff.
            clear_gpu_cache (bool): Default False retains CuPy FFT plans and
                unused memory-pool blocks for reuse. True releases unused cache
                storage between FFT stages and before ASTRA output allocation.
                Later calls repeat FFT setup and allocation. True requires 3D.
            legacy_scaling (bool): Default False uses quantitative scaling.
                True restores the previous SINC global gain and native-width
                FFT. Only supported for 3D SINC. To reproduce the historical
                SINC response, also set ``cutoff_freq=1.1``.
            data_axes_labels_order (list[str] or tuple[str, ...] or None): Legacy
                alias for ``input_axes``, default None. Conflicting axis orders
                raise ValueError.
            filter_parameter (float or None): ASTRA FilterParameter for 2D
                filters that use it. Default None leaves ASTRA's default in
                effect. Unused in 3D.
            filter_d (float or None): ASTRA FilterD for 2D filters that use it.
                Default None leaves ASTRA's default in effect. Unused in 3D.
            recon_mask_radius (float or None): Circular mask size parameter.
                Default None applies no mask; 1.0 selects the inscribed circle.
                Smaller values shrink the mask and larger values expand it.

        Returns:
            numpy.ndarray or cupy.ndarray: Float32 reconstruction using the input
            array library and the requested volume geometry shape. Axis order
            is (Y, X) in 2D and (Z, Y, X) in 3D.

        Notes:
            Quantitative 3D SINC and RAM-LAK use a zero-padded discrete ramp
            with pi/view_count angular normalization. This assumes uniformly
            sampled half-turn views and unit spacing. SINC adds a window with
            unit low-frequency gain; its cutoff controls smoothing. General
            projection vectors and laminography require iterative reconstruction.
        """
        if (
            isinstance(self.projection_geometry, Parallel3DGeometry)
            and not self.projection_geometry.is_standard_tomography
        ):
            raise ValueError(
                "This analytical method requires standard Z-axis tomography; use iterative reconstruction for general vectors or laminography"
            )
        self._ensure_open()
        _boolean(clear_gpu_cache, "clear_gpu_cache")
        _boolean(legacy_scaling, "legacy_scaling")
        if clear_gpu_cache and self.geom != "3D":
            raise ValueError("clear_gpu_cache is supported only for 3D FBP")
        cupyrun = not isinstance(data, np.ndarray) and isinstance(data, cp.ndarray)
        kwargs["cupyrun"] = cupyrun
        selected_filter = (
            filter_type
            if filter_type is not None
            else ("ram-lak" if self.geom == "2D" else "SINC")
        )
        if isinstance(selected_filter, str) and selected_filter.lower() in (
            "ram-lak",
            "ramp",
        ):
            selected_filter = "RAM-LAK" if self.geom == "3D" else "ram-lak"
        if self.geom == "3D" and selected_filter not in ("SINC", "RAM-LAK"):
            raise ValueError("3D FBP filter_type must be 'SINC' or 'RAM-LAK'")
        if self.geom == "2D" and selected_filter == "SINC":
            raise ValueError("SINC requires 3D geometry")
        if legacy_scaling and (self.geom != "3D" or selected_filter != "SINC"):
            raise ValueError("legacy_scaling is supported only for 3D SINC FBP")
        if self.geom == "3D":
            if selected_filter == "SINC":
                cutoff_freq = validate_sinc_cutoff(
                    SINC_DEFAULT_CUTOFF if cutoff_freq is None else cutoff_freq
                )
            else:
                cutoff_freq = validate_ramlak_cutoff(
                    1.0 if cutoff_freq is None else cutoff_freq
                )
        data = self._prepare_projection(
            data,
            array_module=cp,
            input_axes=input_axes,
            legacy_axes=kwargs.get("data_axes_labels_order"),
            gpu=self.geom == "3D",
            pack=True,
            target_axes=(("angles", "detY", "detX") if self.geom == "3D" else None),
        )
        if self.geom == "2D":
            # ASTRA's algorithm/data API takes host storage, unlike direct_BP.
            self.Atools.fbp_filter_type = selected_filter
            self.Atools.fbp_filter_parameter = kwargs.get("filter_parameter")
            self.Atools.fbp_filter_d = kwargs.get("filter_d")
            host_data = cp.asnumpy(data) if cupyrun else data
            reconstruction = self.Atools._fbp(
                np.ascontiguousarray(host_data, dtype=np.float32)
            )
            if cupyrun:
                reconstruction = cp.asarray(reconstruction)
        else:
            # The CUDA filter normalises by its first (angles) dimension.
            apply_filter = (
                _filtersinc3D if selected_filter == "SINC" else _filterramlak3D
            )
            options = (
                {"legacy_scaling": legacy_scaling} if selected_filter == "SINC" else {}
            )
            filtered = apply_filter(
                data, cutoff=cutoff_freq, clear_gpu_cache=clear_gpu_cache, **options
            )
            filtered = cp.ascontiguousarray(filtered.swapaxes(0, 1))
            del data
            if clear_gpu_cache:
                _clear_gpu_cache(cp)
            reconstruction = self.Atools.backproject(filtered)
            if not cupyrun:
                reconstruction = cp.asnumpy(reconstruction)
        return check_kwargs(reconstruction, **kwargs)

    def plan_fourier(
        self,
        input_shape: tuple[int, int, int] | None = None,
        *,
        options: FourierOptions | None = None,
        input_dtype: np.dtype | type | str = np.float32,
    ) -> FourierPlan:
        """Build a Fourier inversion plan without allocating CUDA resources.

        Args:
            input_shape (tuple[int, int, int] or None): Projection shape in canonical
                (detY, angles, detX) order. Default None uses the acquisition geometry.

        Keyword Args:
            options (FourierOptions or None): Validated Fourier settings. Default None
                uses FourierOptions(), including quantitative scaling.
            input_dtype (numpy.dtype or type or str): Input storage dtype used for
                memory accounting, default numpy.float32. Execution packs to float32.

        Returns:
            FourierPlan: Immutable dimensions, chunks and workspace specifications.
            Geometry determines the valid input and output shapes. No CUDA arrays,
            kernels or FFT plans are created; only 3D tomography is supported.
        """
        self._ensure_open()
        return build_fourier_plan(
            self.projection_geometry,
            self.volume_geometry,
            (
                self.projection_geometry.projection_shape
                if input_shape is None
                else input_shape
            ),
            options=options,
            detector_padding=self.detector_padding,
            input_dtype=input_dtype,
        )

    @_on_device(cp)
    def FOURIER_INV(
        self,
        data: np.ndarray | cp.ndarray | tuple[int, int, int],
        *,
        input_axes: list[str] | tuple[str, ...] | None = None,
        options: FourierOptions | None = None,
        **kwargs,
    ) -> cp.ndarray | tuple[int, int, int]:
        """Reconstruct 3D projections by NUFFT inversion.

        Args:
            data (numpy.ndarray or cupy.ndarray or tuple[int, int, int]): Measured
                projections, or a projection shape for memory estimation. Default
                order is (detY, angles, detX). Real arrays are converted to float32
                on the selected GPU.

        Keyword Args:
            input_axes (list[str] or tuple[str, ...] or None): Axis labels for the
                array or shape tuple. Default None uses (detY, angles, detX).
            options (FourierOptions or None): Fourier settings. Default None uses
                FourierOptions(). Supply either this object or individual Fourier
                keyword options; combining both raises TypeError.
            data_axes_labels_order (list[str] or tuple[str, ...] or None): Legacy alias
                for input_axes, default None. Conflicting aliases raise ValueError.
            data_dtype (numpy.dtype or type or str): Input storage dtype for shape-only
                memory estimation, default numpy.float32. Unused with array inputs.
            **kwargs: Individual FourierOptions fields with their declared types and
                defaults. Examples: filter_type="shepp", cutoff_freq=1.0,
                center_size=32768, legacy_scaling=False. None-valued keyword options
                retain defaults; unknown names raise TypeError.

        Returns:
            cupy.ndarray or tuple[int, int, int]: Float32 (Z, Y, X) reconstruction
            matching volume_geometry.shape. For shape-only memory estimation inside
            DeviceMemStack, returns the reconstruction shape instead of an array.

        Notes:
            FFT work-area estimation still requires CUDA; plan_fourier() exposes
            dimensions, chunks and workspace shapes without GPU work.
            Quantitative normalization is the default. legacy_scaling=True restores
            the former Fourier amplitude (8/pi times the physical scale) and odd-width
            crop. See FourierOptions for all execution settings and their defaults.

            Based on [NIKITIN2017]_ and Viktor Nikitin's radonusfft/TomoCuPy implementation.
        """
        self._ensure_open()
        legacy_axes = kwargs.pop("data_axes_labels_order", None)
        data_dtype = kwargs.pop("data_dtype", np.float32)
        kwargs.pop("cupyrun", None)  # Legacy internal output-library flag.
        settings = resolve_fourier_options(options, kwargs)
        axes = _resolve_data_axes(input_axes, legacy_axes, ("detY", "angles", "detX"))
        shape_only = isinstance(data, tuple)
        if not shape_only and not isinstance(data, (np.ndarray, cp.ndarray)):
            raise TypeError(
                "data must be a NumPy/CuPy array or an estimator shape tuple"
            )
        shape = _transpose_data_axes(
            data if shape_only else data.shape,
            source_axes=axes,
            target_axes=("detY", "angles", "detX"),
        )
        plan = self.plan_fourier(
            shape,
            options=settings,
            input_dtype=data_dtype if shape_only else np.float32,
        )
        executor = FourierExecutor(self.projection_geometry, plan)
        if shape_only:
            stack = DeviceMemStack.instance()
            if stack is None:
                raise ValueError(
                    "shape-only Fourier estimation requires a DeviceMemStack context"
                )
            return executor.estimate(stack)
        data = self._prepare_projection(
            data,
            array_module=cp,
            input_axes=axes,
            gpu=True,
            pack=True,
            pad_detector=False,
        )
        return executor.execute(data)
