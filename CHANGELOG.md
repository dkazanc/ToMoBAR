# Changelog
All notable changes to this project will be documented in this file.

## [2020.01-2020.06]
### Added
- Stripe-Weighted Least squares penalty to remove ring artifacts
- Mask initialisation to apply circular masking to the reconstructed image/VOLUME
- An option to provide variable CoRs values in order to fix the problem of misalignment for 2D and 3D case
- Additional real data demo to reconstruct macrocrystallographic data
- Various bug fixes

## [2020.01]
### Added
- Kullback-Leibler term has been added according to C.Vogel p.174
- Demos has been updated to use KL term

## [2019.12]
### Changed
- Due to model-based structure of algorithms, the amount of parameters constantly increases. It has been
decided to re-structure class function and make it simpler to operate. See demos for examples.
- Added extensive HELP, one can use - help(RecToolsIR) or help(RecToolsDIR) to get information about parameters
- Demos modified to adapt a new structure

## [2019.11]

### Added
- RING_WEIGHTS module in C which calculates a better ring model to use in non-quadratic data penalties
- Cmake and Cython wrappers
- run.sh to run Cmake based installation of Python wrappers

### Changed
- Installation process to use Cmake and Cython to wrap Python modules
- Demos and conda-build files moved to the main directory

## [2019.08]
### Added
- Autocropper to automatically crop the 3D projection data to reduce its size

### Changed
- normalisation script has been optimised

## [2019.06]
### Added
- Vector geometry added to 3D case replacing the scalar approach
- Center of Rotation (CenterRotOffset variable) can be defined in the class as a scalar to avoid direct data manipulations (cropping, padding)

### Changed
- Demos for 2D/3D reconstruction updated with a new class

### Removed
- Section about "changelog" vs "CHANGELOG".
