| Master | Anaconda binaries |
|--------|-------------------|
| [![Build Status](https://travis-ci.org/dkazanc/ToMoBAR.svg?branch=master)](https://travis-ci.org/dkazanc/ToMoBAR.svg?branch=master) | ![conda version](https://anaconda.org/dkazanc/tomobar/badges/version.svg) ![conda last release](https://anaconda.org/dkazanc/tomobar/badges/latest_release_date.svg) [![conda platforms](https://anaconda.org/dkazanc/tomobar/badges/platforms.svg) ![conda dowloads](https://anaconda.org/dkazanc/tomobar/badges/downloads.svg)](https://anaconda.org/dkazanc/tomobar/) |

# TOmographic MOdel-BAsed Reconstruction (ToMoBAR)

## ToMoBAR is a library of direct and model-based regularised iterative reconstruction algorithms with a *plug-and-play* capability

### Software includes:
 ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) A wrapper around [ASTRA-toolbox](https://www.astra-toolbox.com/) to simplify access to various reconstruction methods

 ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Regularised iterative ordered-subsets [FISTA](https://epubs.siam.org/doi/10.1137/080716542) reconstruction algorithm with linear and non-linear data fidelities

 ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Regularised iterative [ADMM](https://ieeexplore.ieee.org/document/7744574/) reconstruction algorithm

 ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Demos to reconstruct synthetic and also real data (provided) [4-6]

<div align="center">
  <img src="docs/images/recsFISTA_stud.png" height="216"><br>  
</div>
<div align="center">
  <img src="docs/images/TomoRec_surf2.jpg" height="250"><br>  
</div>

## Software highlights:
 * Tomographic projection data can be simulated without the "inverse crime" using [TomoPhantom](https://github.com/dkazanc/TomoPhantom). Noise and artifacts (zingers, rings, jitter) can be modelled and added to the data.
 * Simulated data reconstructed iteratively using FISTA or ADMM algorithms with multiple "plug-and-play" regularisers from [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit).
 * The FISTA algorithm offers various modifications: convergence acceleration with ordered-subsets method, PWLS, Huber, Group-Huber[3] and Students't data fidelities [1,2] to deal with noise and imaging artifacts (rings, streaks).

### General software prerequisites
 * [MATLAB](http://www.mathworks.com/products/matlab/) or Python

### Software dependencies:
 * [ASTRA-toolbox](https://www.astra-toolbox.com/) for projection operations
 * [TomoPhantom](https://github.com/dkazanc/TomoPhantom) for simulation
 * [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) for regularisation
 * See [INSTALLATION](https://github.com/dkazanc/TomoRec/blob/master/INSTALLATION) for detailed information

### Installation (Python or Matlab)

#### Python standalone
For building on Linux see `run.sh`

#### Python conda:
Install from the conda channel:
```
conda install -c dkazanc tomobar
```
 or build with:
```
export VERSION=`date +%Y.%m` (unix) / set VERSION=2019.11 (Windows)
conda build conda-recipe/ --numpy 1.16 --python 3.6
conda install tomobar --use-local --force-reinstall # if this don't work (probably you're on Python 3*)
conda install -c file://${CONDA_PREFIX}/conda-bld/ tomobar --force-reinstall # try this one
```
#### Matlab:
Simply use available m-functions, see Demos

### References:
 1. [D. Kazantsev et al. 2017. A Novel Tomographic Reconstruction Method Based on the Robust Student's t Function For Suppressing Data Outliers. IEEE TCI, 3(4), pp.682-693.](https://doi.org/10.1109/TCI.2017.2694607)
 2. [D. Kazantsev et al. 2017. Model-based iterative reconstruction using higher-order regularization of dynamic synchrotron data. Measurement Science and Technology, 28(9), p.094004.](https://doi.org/10.1088/1361-6501/aa7fa8)
 3. [P. Paleo and A. Mirone, 2015. Ring artifacts correction in compressed sensing tomographic reconstruction. Journal of synchrotron radiation, 22(5), pp.1268-1278.](https://doi.org/10.1107/S1600577515010176)

### Applications (where ToMoBAR software have been used):
 4. [D. Kazantsev et al. 2019. CCPi-Regularisation toolkit for computed tomographic image reconstruction with proximal splitting algorithms. SoftwareX, 9, pp.317-323.](https://doi.org/10.1016/j.softx.2019.04.003)
 5. [E. Guo et al. 2018. The influence of nanoparticles on dendritic grain growth in Mg alloys. Acta Materialia.](https://doi.org/10.1016/j.actamat.2018.04.023)
 6. [E. Guo et al. 2018. Revealing the microstructural stability of a three-phase soft solid (ice cream) by 4D synchrotron X-ray tomography. Journal of Food Engineering](https://www.sciencedirect.com/science/article/pii/S0260877418302309)
 7. [E. Guo et al. 2017. Dendritic evolution during coarsening of Mg-Zn alloys via 4D synchrotron tomography. Acta Materialia](https://doi.org/10.1016/j.actamat.2016.10.022)
 8. [E. Guo et al. 2017. Synchrotron X-ray tomographic quantification of microstructural evolution in ice cream–a multi-phase soft solid. Rsc Advances](https://doi.org/10.1039/C7RA00642J)

### License:
GNU GENERAL PUBLIC LICENSE v.3

### Questions/Comments
can be addressed to Daniil Kazantsev at dkazanc@hotmail.com
