"""Isolated ASTRA/CuPy backends shared by pytest fixtures and unittest cases.

Dependency patches exist only while loading isolated production module objects.
Each harness owns an ExitStack for reconstructor/projector cleanup.
"""

import importlib.util
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)

ROOT = Path(__file__).resolve().parents[2] / "tomobar"


def stub_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class _Harness:
    def close(self):
        self._cleanup.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class MockGPUArray:
    """NumPy-backed storage for testing GPU metadata and packing without CUDA."""

    def __init__(self, storage, device=0):
        self.storage = storage
        self.device = SimpleNamespace(id=device)
        self.shape = storage.shape
        self.ndim = storage.ndim
        self.dtype = storage.dtype
        self.itemsize = storage.itemsize
        self.flags = storage.flags
        self.data = SimpleNamespace(ptr=storage.ctypes.data)


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MockAstra:
    """Strict ID registries catch wrong-manager and double deletion."""

    def __init__(self):
        self.live = {
            name: set()
            for name in ("data2d", "data3d", "projector", "projector3d", "algorithm")
        }
        self.next_id = 1
        self.counts = {}
        self.fail_at = None
        self.configs = {}
        self.deleted = []
        self.result = np.ones((2, 2), dtype=np.float32)
        self.module = ModuleType("astra")
        for name in self.live:
            manager = SimpleNamespace(
                create=Mock(
                    side_effect=lambda *a, _n=name: self.allocate(_n, "create", *a)
                ),
                link=Mock(
                    side_effect=lambda *a, _n=name: self.allocate(_n, "link", *a)
                ),
                delete=Mock(side_effect=lambda ident, _n=name: self.delete(_n, ident)),
                get=Mock(side_effect=lambda ident, _n=name: self.get(_n, ident)),
            )
            setattr(self.module, name, manager)
        self.module.algorithm.run = Mock(side_effect=self.run)
        self.module.set_gpu_index = Mock()
        self.module.astra_dict = Mock(side_effect=lambda method: {"type": method})
        self.module.create_vol_geom = Mock(return_value="volume")
        self.module.create_proj_geom = Mock(side_effect=lambda *a: object())
        self.module.geom_size = Mock(return_value=(2, 3, 4))
        self.module.create_projector = Mock(side_effect=self.create_projector)

    def hit(self, event):
        self.counts[event] = self.counts.get(event, 0) + 1
        if self.fail_at == (event, self.counts[event]):
            raise RuntimeError("injected failure")

    def allocate(self, manager, operation, *args):
        self.hit(f"{manager}.{operation}")
        ident = self.next_id
        self.next_id += 1
        self.live[manager].add(ident)
        if manager == "algorithm":
            cfg = args[0]
            if "ProjectorId" in cfg:
                assert cfg["ProjectorId"] in self.live["projector"]
            self.configs[ident] = cfg
        return ident

    def create_projector(self, kind, *args, **kwargs):
        return self.allocate(
            "projector3d" if kind == "cuda3d" else "projector", "create"
        )

    def delete(self, manager, ident):
        assert ident in self.live[manager], (
            "wrong manager or double deletion",
            manager,
            ident,
        )
        if manager != "algorithm":
            for alg in self.live["algorithm"]:
                cfg = self.configs[alg]
                for key in (
                    "ProjectorId",
                    "VolumeDataId",
                    "ProjectionDataId",
                    "ReconstructionDataId",
                ):
                    assert (
                        cfg.get(key) != ident
                    ), "delete algorithms before their dependencies"
        self.live[manager].remove(ident)
        self.deleted.append((manager, ident))

    def get(self, manager, ident):
        self.hit(f"{manager}.get")
        assert ident in self.live[manager]
        return self.result

    def run(self, ident, *args):
        self.hit("algorithm.run")
        assert ident in self.live["algorithm"]


class ResourceHarness(_Harness):
    def __init__(self):
        self._cleanup = ExitStack()
        self.backend = MockAstra()
        self.astra = self.backend.module
        self.array = MockGPUArray(np.ones((2, 3, 4), dtype=np.float32))
        self.output = None
        self.current_device = 0
        cupy = ModuleType("cupy")
        cupy.ndarray = MockGPUArray
        cupy.float32 = np.float32
        owner = self

        class Device:
            compute_capability = "80"

            def __init__(self, index):
                self.index = index

            def __enter__(self):
                self.previous = owner.current_device
                owner.current_device = self.index
                return self

            def __exit__(self, *exc):
                owner.current_device = self.previous

        cupy.cuda = SimpleNamespace(Device=Mock(side_effect=Device))
        cupy.ascontiguousarray = Mock(side_effect=self.ascontiguousarray)
        cupy.empty = Mock(side_effect=self.empty)
        self.cupy = cupy
        self.astra.projector.direct_FP = Mock(side_effect=self.direct)
        self.astra.projector.direct_BP = Mock(side_effect=self.direct)
        self.astra.projector3d.direct_FP = Mock(side_effect=self.direct)
        self.astra.projector3d.direct_BP = Mock(side_effect=self.direct)
        with patch.dict("sys.modules", {"astra": self.astra, "cupy": cupy}):
            axes = load_module("_resource_test_axes", "supp/axis_order.py")
            self.axes = axes
            with patch.dict("sys.modules", {"tomobar.supp.axis_order": axes}):
                self.base = load_module(
                    "_resource_test_base", "astra_wrappers/astra_base.py"
                )
                with patch.dict(
                    "sys.modules", {"tomobar.astra_wrappers.astra_base": self.base}
                ):
                    self.tools2d = load_module(
                        "_resource_test_2d", "astra_wrappers/astra_tools2d.py"
                    )
                    self.tools3d = load_module(
                        "_resource_test_3d", "astra_wrappers/astra_tools3d.py"
                    )

    def ascontiguousarray(self, data, dtype):
        self.backend.hit("ascontiguousarray")
        assert self.current_device == data.device.id
        packed = np.ascontiguousarray(data.storage, dtype=dtype)
        result = (
            data if packed is data.storage else MockGPUArray(packed, data.device.id)
        )
        self.prepared = weakref.ref(result)
        return result

    def empty(self, shape, dtype):
        self.backend.hit("empty")
        self.output = MockGPUArray(np.empty(shape, dtype=dtype), self.current_device)
        return self.output

    def direct(self, ident, data, *, out):
        self.backend.hit("direct")
        assert (
            ident in self.backend.live["projector3d"] | self.backend.live["projector"]
        )
        if isinstance(data, np.ndarray):
            assert data.flags.c_contiguous
            assert data.dtype == np.float32
            out[:] = 1.0
            return
        prepared = self.prepared()
        assert prepared is not None, "normalized storage must survive the direct call"
        assert prepared.flags.c_contiguous
        assert prepared.dtype == np.float32
        self.projected_input = prepared.storage.copy()
        assert prepared.device.id == self.current_device
        assert self.output.device.id == self.current_device
        assert data is prepared
        assert out is self.output
        assert self.astra.create_projector.call_args.kwargs["options"] == {
            "GPUindex": self.current_device
        }

    def make_wrapper(
        self,
        dim=2,
        subsets=1,
        register_cleanup=True,
        offset=0.0,
        angles=None,
        detector_shifts=None,
    ):
        common = dict(
            detectors_x=4,
            detectors_x_pad=0,
            angles_vec=np.arange(4.0) if angles is None else angles,
            centre_of_rotation=offset,
            recon_size=2,
            device_index=0,
            ordsub_number=subsets,
        )
        if dim == 2:
            wrapper = self.tools2d.AstraTools2D(
                projection_geometry=Parallel2DGeometry(
                    angles=common["angles_vec"],
                    detector_shape=(4,),
                    centre_of_rotation=offset,
                    detector_shifts=detector_shifts,
                ),
                volume_geometry=Volume2DGeometry(shape=(2, 2)),
                device_index=0,
                ordsub_number=subsets,
            )
        else:
            wrapper = self.tools3d.AstraTools3D(
                projection_geometry=Parallel3DGeometry.tomography(
                    angles=common["angles_vec"],
                    detector_shape=(2, 4),
                    centre_of_rotation=offset,
                    detector_shifts=detector_shifts,
                ),
                volume_geometry=Volume3DGeometry(shape=(2, 2, 2)),
                device_index=0,
                ordsub_number=subsets,
            )
        if register_cleanup:
            self._cleanup.callback(wrapper.close)
        return wrapper

    def load_iterative_module(self):
        """Run actual iterative control flow with NumPy and mocked projectors."""
        cupy = ModuleType("cupy")
        cupy.__getattr__ = lambda name: getattr(np, name)
        cupy.random = SimpleNamespace(
            randn=lambda *shape, dtype: np.ones(shape, dtype=dtype)
        )
        from contextlib import nullcontext

        cupy.cuda = SimpleNamespace(
            Device=Mock(side_effect=lambda index: nullcontext())
        )
        cupy._default_memory_pool = SimpleNamespace(free_all_blocks=Mock())
        supp = ModuleType("suppTools")
        supp.check_kwargs = lambda data, **kw: data
        supp.perform_recon_crop = lambda data, *args: data
        regularisers = ModuleType("regularisers")
        regularisers.prox_regul = Mock()
        fidelities = ModuleType("fidelities")
        fidelities.grad_data_term = Mock(
            side_effect=lambda rec, x, *args, **kwargs: np.zeros_like(x)
        )
        import importlib

        importlib.import_module("tomobar.solver_state")
        with patch.dict(
            "sys.modules",
            {
                "cupy": cupy,
                "astra": self.astra,
                "tomobar.supp.axis_order": self.axes,
                "tomobar.supp.suppTools": supp,
                "tomobar.regularisers": regularisers,
                "tomobar.astra_wrappers.astra_tools3d": self.tools3d,
                "tomobar.astra_wrappers.astra_tools2d": self.tools2d,
                "tomobar.data_fidelities": fidelities,
            },
        ):
            operators = load_module("_subset_test_operators", "operators.py")
            with patch.dict("sys.modules", {"tomobar.operators": operators}):
                return load_module("_subset_test_ir", "methodsIR.py")

    def make_iterative(self, module, n_views, count):
        rec = module.RecToolsIR(
            projection_geometry=Parallel3DGeometry(
                angles=np.arange(float(n_views)), detector_shape=(1, 2)
            ),
            volume_geometry=Volume3DGeometry(shape=(1, 2, 2)),
            OS_number=count,
        )
        self._cleanup.callback(rec.close)
        return rec


class DirectHarness(_Harness):
    def __init__(self):
        self._cleanup = ExitStack()
        identity = lambda data, *args, **kwargs: data

        class Device:
            compute_capability = "80"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        cupy = stub_module(
            "cupy",
            ndarray=np.ndarray,
            RawKernel=Mock,
            float32=np.float32,
            ascontiguousarray=np.ascontiguousarray,
            swapaxes=np.swapaxes,
            asnumpy=np.asarray,
            asarray=np.asarray,
            cuda=SimpleNamespace(Device=Mock(return_value=Device())),
            fft=SimpleNamespace(
                config=SimpleNamespace(
                    get_plan_cache=Mock(return_value=SimpleNamespace(clear=Mock()))
                )
            ),
            _default_memory_pool=SimpleNamespace(free_all_blocks=Mock()),
        )
        self.filter = Mock(side_effect=identity)
        self.kernels = Mock(side_effect=RuntimeError("stop before CUDA kernels"))
        executor = Mock()
        executor.execute.side_effect = RuntimeError("stop before CUDA kernels")
        executor.estimate.side_effect = RuntimeError("stop before CUDA kernels")
        stubs = {
            "tomobar._fourier_execution": stub_module(
                "fourier_execution", FourierExecutor=Mock(return_value=executor)
            ),
            "cupy": cupy,
            "scipy": stub_module("scipy"),
            "scipy.fftpack": stub_module("scipy.fftpack"),
            "cupyx.scipy.fft": stub_module(
                "cupyx.scipy.fft",
                **{
                    name: Mock()
                    for name in ("fft", "ifft2", "rfftfreq", "rfft", "irfft")
                },
            ),
            "cupyx.scipy.fftpack": stub_module(
                "cupyx.scipy.fftpack", get_fft_plan=Mock()
            ),
            "tomobar.astra_wrappers.astra_tools2d": stub_module(
                "astra_tools2d", AstraTools2D=Mock()
            ),
            "tomobar.astra_wrappers.astra_tools3d": stub_module(
                "astra_tools3d", AstraTools3D=Mock()
            ),
            "tomobar.supp.suppTools": stub_module("suppTools", check_kwargs=identity),
            "tomobar.supp.memory_estimator_helpers": stub_module(
                "memory", DeviceMemStack=Mock()
            ),
            "tomobar.fourier": stub_module(
                "fourier",
                _filtersinc3D=self.filter,
                _filterramlak3D=self.filter,
                calc_filter=Mock(),
            ),
            "tomobar.cuda_kernels": stub_module(
                "cuda_kernels",
                load_cuda_module=self.kernels,
                get_cuda_kernel=lambda module, name: module.get_function(name),
            ),
        }
        with patch.dict("sys.modules", stubs):
            self.axes = load_module("_axis_test_axes", "supp/axis_order.py")
            with patch.dict("sys.modules", {"tomobar.supp.axis_order": self.axes}):
                operators = load_module("_axis_test_operators", "operators.py")
                with patch.dict("sys.modules", {"tomobar.operators": operators}):
                    self.direct_module = load_module("_axis_test_dir", "methodsDIR.py")
        self.classes = (self.direct_module.RecToolsDIR,)

    def reconstructor(self, cls, native, dim=3):
        rec = cls.__new__(cls)
        if dim == 3:
            scan = Parallel3DGeometry(
                angles=np.arange(native.shape[1], dtype=float),
                detector_shape=(native.shape[0], native.shape[-1]),
            )
            volume = Volume3DGeometry(
                shape=(native.shape[0], native.shape[-1], native.shape[-1])
            )
        else:
            scan = Parallel2DGeometry(
                angles=np.arange(native.shape[0], dtype=float),
                detector_shape=(native.shape[-1],),
            )
            volume = Volume2DGeometry(shape=(native.shape[-1], native.shape[-1]))
        rec._init_device(0)
        rec._init_geometry(scan, volume, 0)
        rec.Atools = SimpleNamespace(
            close=Mock(),
            forward=Mock(return_value=native),
            backproject=Mock(side_effect=lambda data: data),
            _fbp=Mock(side_effect=lambda data: data),
            detectors_x_pad=0,
        )
        self._cleanup.callback(rec.close)
        return rec
