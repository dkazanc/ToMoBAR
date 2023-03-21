% adding paths
fsep = '/';
packagename = 'tomobar'
% TomoPhantom paths (TomoPhantom needs to be compiled first)
pathtoTomoPhantom = sprintf(['..' fsep '..' fsep packagename fsep 'matlab' fsep 'supplementary' fsep 'TomoPhantom' fsep 'Wrappers' fsep 'MATLAB' fsep 'compiled' fsep], 1i);
addpath(pathtoTomoPhantom);
pathtoTomoPhantom2 = sprintf(['..' fsep '..' fsep packagename fsep 'matlab' fsep  'supplementary' fsep 'TomoPhantom' fsep 'Wrappers' fsep 'MATLAB' fsep 'supplem' fsep], 1i);
addpath(pathtoTomoPhantom2);
pathtoModels = sprintf(['..' fsep '..' fsep packagename fsep 'matlab' fsep 'supplementary' fsep 'TomoPhantom' fsep 'PhantomLibrary' fsep 'models' fsep 'Phantom2DLibrary.dat'], 1i);
% Regularisation Toolkit path to compiled library (CCPi-RGTtk needs to be compiled first)
pathtoRGLtk = sprintf(['..' fsep '..' fsep packagename fsep 'matlab' fsep 'supplementary' fsep 'CCPi-Regularisation-Toolkit' fsep 'Wrappers' fsep 'Matlab' fsep 'mex_compile' fsep 'installed'], 1i);
addpath(pathtoRGLtk);

pathmainFunc = sprintf(['..' fsep '..' fsep packagename fsep 'matlab' fsep 'main_func'], 1i);
addpath(pathmainFunc); 
pathSupp = sprintf(['..' fsep '..' fsep packagename fsep 'matlab' fsep 'supplementary'], 1i);
addpath(pathSupp);