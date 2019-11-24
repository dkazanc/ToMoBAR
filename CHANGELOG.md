# Changelog
All notable changes to this project will be documented in this file.

## [2019.12]
### Changed
- Due to model-based structure of algorithms, the amount of parameters increases. It has been 
decided to restructure class function and make it simpler to operate. See demos.
- Added extensive HELP, one can use - help(RecToolsIR) or help(RecToolsDIR) to get information about all parameters
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
- Vector geometry added to 3D case replacing the scalar aproach
- Center of Rotation (CenterRotOffset variable) can be defined in the class as a scalar to avoid direct data manupulations (cropping, padding)

### Changed
- Demos for 2D/3D reconstruction updated with a new class

### Removed
- Section about "changelog" vs "CHANGELOG".
