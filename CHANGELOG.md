# Change Log
All notable changes in ToMoBAR project will be documented in this file.

## v.1.0

To better communicate breaking changes, ToMoBAR is moving from calendar versioning to semantic versioning. The initial `1.0` release is based on the `2025.12` version.

## v.2025.10

### Changed
`DetectorsDimH_pad` parameter works now in CuPy iterative methods available in `RecToolsIRCuPy` class.
This allows reconstructing without artifacts on the edges of the reconstruction grid. Especially useful for
reconstructing data that is larger than the field of view.

### Fixed
SIRT CuPy algorithm has been fixed.

### New
FISTA iterative method in `RecToolsIRCuPy` dropped its dependency on Regularisation Toolkit. Currently two regularisers
available to use: `PD_TV` and `ROF_TV` and they were accelerated and optimised. Therefore FISTA with CuPy is up to 3 times faster compared
to the previous version and potentially a magnitude faster compared to FISTA in `RecToolsIR`.

## v.2025.08

### Changed
- $\sf\color{red}Breaking$ $\sf\color{red}changes!$ The API for initializing geometry in both direct and iterative methods (the `RecTools` class) has been updated. A new parameter, `DetectorsDimH_pad`, has been [introduced](https://dkazanc.github.io/ToMoBAR/api/tomobar.methodsDIR.html) to control edge padding along the detector's horizontal dimension. This parameter can help reduce circular/arc [artifacts](https://dkazanc.github.io/ToMoBAR/tutorials/real_data_recon.html) in reconstructions, such as saturated circles or arcs. See updated [Tutorials](https://dkazanc.github.io/ToMoBAR/tutorials/direct_recon.html) and [Demos](https://github.com/dkazanc/ToMoBAR/tree/master/Demos/Python).
- Log-Polar method (`FOURIER_INV` in `RecToolsDIRCuPy`) has been further accelerated and it is significantly faster FBP.

