Installation troubleshooting
============================

Start with the checks in :doc:`installation`. Use this page when imports,
CUDA execution or notebook kernels do not behave as expected.

Common problems
---------------

* **Missing CuPy or ASTRA:** check the ``sys.executable`` output above. Activate
  the reconstruction environment or select its Jupyter kernel; the
  ``tomobar-docs`` environment is for documentation and CPU geometry plotting.
* **No CUDA device:** check the NVIDIA driver and the compatibility of the CUDA
  packages selected for CuPy and ASTRA. An ASTRA CPU-only build does not provide
  ToMoBAR reconstruction.
* **Missing CUDA headers or compilation errors:** follow the CUDA headers and
  custom-installation guidance in the
  `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_.
  Device detection alone does not exercise kernel compilation; use the array,
  FFT and reconstruction checks above.
* **Imports or arguments from an older release:** follow
  :doc:`../howto/migration` and check which ToMoBAR version is installed.

Detailed GPU verification
-------------------------

Run these checks in the same environment and Python kernel that will execute
ToMoBAR. In Python or a notebook cell::

    import sys
    from importlib.metadata import version

    import astra
    import cupy as cp
    from tomobar.reconstruction.direct import DirectReconstructor
    from tomobar.reconstruction.iterative import IterativeReconstructor

    print("Python:", sys.executable)
    print("ToMoBAR:", version("tomobar"))
    print("CUDA devices:", cp.cuda.runtime.getDeviceCount())
    assert astra.use_cuda(), "ASTRA cannot use CUDA in this environment"

    signal = cp.arange(16, dtype=cp.float32)
    restored = cp.fft.irfft(cp.fft.rfft(signal), n=signal.size)
    cp.testing.assert_allclose(restored, signal, atol=1e-5)
    print("CuPy array and FFT checks passed")

The device count should be positive and both reconstruction imports should
succeed. These checks exercise basic CUDA array operations and FFTs. Complete
:doc:`../tutorials/synthetic_3d` to verify projection, filtering and reconstruction
together. Use :doc:`first_reconstruction` to follow the learning path.

Notebook environments and kernels
---------------------------------

In your reconstruction environment, install the plotting and notebook tools::

    python -m pip install matplotlib jupyterlab ipykernel

Download a notebook from :doc:`../tutorials/index`, or launch JupyterLab from
the repository root to open the maintained notebook sources::

    python -m jupyterlab docs/source/tutorials

Select the kernel for the environment in which reconstruction works. If you
need to expose that environment to JupyterLab running elsewhere, register its
kernel from the reconstruction environment::

    python -m ipykernel install --user --name tomobar --display-name 'Python (ToMoBAR)'

The introductory 3D and regularised notebooks generate their inputs. TomoPhantom and
experimental datasets are not needed. Run all cells and save the outputs to
refresh the figures published in the documentation; see
:doc:`../developers/examples`.
