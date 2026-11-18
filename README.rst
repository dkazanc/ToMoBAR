ToMoBAR
========

ToMoBAR is a Python library for GPU-accelerated direct and model-based iterative
reconstruction of parallel-beam tomography and laminography data.

It provides FBP, Fourier inversion for supported 3D scans, and the 3D iterative
methods Landweber, SIRT, CGLS, FISTA, ADMM and OSEM. Explicit acquisition and
volume geometry objects define array shapes and coordinate conventions.
ASTRA supports flexible geometry; USFFT supports a restricted set of
angle-based tomography geometries.

FISTA and ADMM support LS, PWLS and SWLS data fidelities. FISTA supports ordered
subsets; ADMM uses full-data updates. Native CuPy regularisers are bundled,
so CCPi-Regularisation-Toolkit is not required.

Reconstruction requires a CUDA-capable NVIDIA GPU. Direct forward projection,
backprojection and FBP preserve the NumPy or CuPy input library; iterative
reconstruction returns CuPy images. Selecting a device supports integration
into larger applications, but ToMoBAR itself does not distribute reconstruction
across multiple GPUs or MPI processes.

Start with the `installation guide
<https://dkazanc.github.io/ToMoBAR/getting_started/installation.html>`_
and the `executed notebook tutorials
<https://dkazanc.github.io/ToMoBAR/tutorials/index.html>`_. The notebooks include
saved figures and need no GPU to read. The geometry tutorial can also run
without a GPU.

See `Readme.md <Readme.md>`_ for package requirements, related software and
citation information, or browse the `standalone demos
<https://github.com/dkazanc/ToMoBAR/tree/master/demos>`_.
