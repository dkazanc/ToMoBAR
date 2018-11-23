#  Regularised FISTA-type iterative reconstruction algorithm for X-ray tomographic reconstruction with highly inaccurate measurements

**This software supports research published in the following journal papers [1,2,3] with applications in [4-6]. Software depends on several software packages and requires a GPU (Nvidia) card to operate. FISTA-tomo is implemented in both MATLAB and Python (work in progress).** 

<div align="center">
  <img src="docs/images/recsFISTA_stud.png" height="216"><br>  
</div>

## Software highlights:
 * Tomographic projection data are simulated without the "inverse crime" using [TomoPhantom](https://github.com/dkazanc/TomoPhantom). Noise and artifacts (zingers, rings) can be modelled and added to data.
 * Simulated data reconstructed iteratively using FISTA-type algorithm with multiple "plug-and-play" regularisers from [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) 
 * Presented FISTA algorithm offers novel modifications: convergence acceleration with ordered-subsets method, PWLS, Group-Huber[3] and Students't data fidelities [1,2] to deal with noise and image artifacts
 * Various projection (2D/3D) geometries are supported and real data provided to demonstrate the effectiveness of the method  

### General software prerequisites
 * [MATLAB](http://www.mathworks.com/products/matlab/) or
 * Python
 * C compilers (GCC/MinGW) and nvcc [CUDA SDK](https://developer.nvidia.com/cuda-downloads) compilers
 
### Software dependencies: 
 * [ASTRA-toolbox](https://www.astra-toolbox.com/)  
 * [TomoPhantom](https://github.com/dkazanc/TomoPhantom)
 * [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) 
 * See [INSTALLATION](https://github.com/dkazanc/FISTA-tomo/blob/master/INSTALLATION) for detailed information

### Installation in Python (conda-build):
```
	conda build Wrappers/Python/conda-recipe --numpy 1.12 --python 3.5 
	conda install fista-tomo --use-local --force

```
 
### Package contents:
 * A number of demos for 2D/3D parallel and cone-beam geometry with 2D and 3D regularisation. Demos show how the methods deal with noise and artifacts. Also real-data example added to emphasise methods properties. Main reconstruction function *FISTA_REC.m* is controlled by various parameters, see the [details](https://github.com/dkazanc/FISTA-tomo/blob/master/main_func/FISTA_REC.m). 

### References:
 1. [D. Kazantsev et al. 2017. A Novel Tomographic Reconstruction Method Based on the Robust Student's t Function For Suppressing Data Outliers. IEEE TCI, 3(4), pp.682-693.](https://doi.org/10.1109/TCI.2017.2694607)
 2. [D. Kazantsev et al. 2017. Model-based iterative reconstruction using higher-order regularization of dynamic synchrotron data. Measurement Science and Technology, 28(9), p.094004.](https://doi.org/10.1088/1361-6501/aa7fa8)
 3. [P. Paleo and A. Mirone, 2015. Ring artifacts correction in compressed sensing tomographic reconstruction. Journal of synchrotron radiation, 22(5), pp.1268-1278.](https://doi.org/10.1107/S1600577515010176)

### Applications:
 4. [E. Guo et al. 2018. The influence of nanoparticles on dendritic grain growth in Mg alloys. Acta Materialia.](https://doi.org/10.1016/j.actamat.2018.04.023) 
 5. [E. Guo et al. 2018. Revealing the microstructural stability of a three-phase soft solid (ice cream) by 4D synchrotron X-ray tomography. Journal of Food Engineering, vol.237](https://www.sciencedirect.com/science/article/pii/S0260877418302309)
 6. [E. Guo et al. 2017. Dendritic evolution during coarsening of Mg-Zn alloys via 4D synchrotron tomography. Acta Materialia, 123, pp.373-382.](https://doi.org/10.1016/j.actamat.2016.10.022) 
 7. [E. Guo et al. 2017. Synchrotron X-ray tomographic quantification of microstructural evolution in ice cream–a multi-phase soft solid. Rsc Advances, 7(25), pp.15561-15573.](https://doi.org/10.1039/C7RA00642J)
 
### License:
GNU GENERAL PUBLIC LICENSE v.3

### Questions/Comments
can be addressed to Daniil Kazantsev at dkazanc@hotmail.com
