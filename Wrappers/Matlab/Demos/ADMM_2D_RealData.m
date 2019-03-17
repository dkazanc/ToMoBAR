% A demo script to reconstruct 2D parallel-beam real-data (Collected at I12 branchline, Diamond Light Source).

% The reconstruction method is the regularised Alternating Direction method of
% Multipliers (ADMM method)

% Requirements: ASTRA-toolbox with GPU-enabled device, SPOT operator, the TomoPhantom package, and the CCPi-Regularisation toolkit. 
% See ReadMe for more information and references

close all;clc;clear;
% adding paths
fsep = '/';

% Regularisation Toolkit path to compiled library (CCPi-RGTtk needs to be compiled first)
pathtoRGLtk = sprintf(['..' fsep 'supplementary' fsep 'CCPi-Regularisation-Toolkit' fsep 'Wrappers' fsep 'Matlab' fsep 'mex_compile' fsep 'installed'], 1i);
addpath(pathtoRGLtk);

pathmainFunc = sprintf(['..' fsep 'main_func'], 1i);
addpath(pathmainFunc); 
pathSupp = sprintf(['..' fsep 'supplementary'], 1i);
addpath(pathSupp);
%%
load ../../../data/RealDataDend.mat
sinoRD = sinoRD';
figure; imshow(sinoRD, [ ]); colormap hot; title('Sinogram');
%%
fprintf('\n %s\n', 'Generating geometry using ASTRA toolbox...');
N = 1000;
% 2D geometry to reconstruct one selected slice
proj_geom = astra_create_proj_geom('parallel', 1.0, Detectors, anglesRad);
vol_geom = astra_create_vol_geom(N,N);

% Create the Spot operator for ASTRA using the GPU.
A = opTomo('cuda', proj_geom, vol_geom);
%%
clear params;
params.A = A; % projection matrix
params.sino = sinoRD(:); % vectorised sinogram
params.iterADMM = 12; % the number of outer ADMM iterations
params.rhoADMM = 500; % Convergence-related ADMM constant
params.alphaADMM = 1; % ADMM variable
params.iterKrylov = 15; % the number of iterations for Krylov method to solve SLAE
params.TolKrylov = 1e-5; % tolerance parameter for Krylov updates

%>>>>>>>>>>>> Regularisation block <<<<<<<<<<<<<<
params.Regul_device = 'gpu'; % select 'cpu' or 'gpu' device for regularisation

% Select preferable regulariser (see more information on CCPi-RGL toolkit).
regulariser = 'ROF_TV'; % Rudin-Osher-Fatemi Total Variation functional 
% regulariser = 'FGP_TV'; % Fast-gradient-projection Total Variation method
% regulariser = 'SB_TV'; % Split Bregman Total Variation method
% regulariser = 'TGV'; % Total Generilised Variation method
% regulariser = 'ROFLLT'; % LLT-ROF model (higher order regulariser)
% regulariser = 'Diff2nd'; % Nonlinear diffusion regulariser
% regulariser = 'Diff4nd'; % Anisotropic diffusion of 4th order 

if (strcmp(regulariser, 'ROF_TV') == 1)
params.Regul_Lambda_ROFTV = 1500; % regularisation parameter
params.Regul_time_step = 0.00025; % time marching parameter (convergence)
params.Regul_Iterations = 3500; % inner iterations number for regularisation
elseif (strcmp(regulariser, 'FGP_TV') == 1)
params.Regul_Lambda_FGPTV = 1500; % regularisation parameter
params.Regul_Iterations = 150; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'SB_TV') == 1)
params.Regul_Lambda_SBTV = 1500; % regularisation parameter
params.Regul_Iterations = 100; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'TGV') == 1)
params.Regul_Lambda_TGV = 3000; % regularisation parameter
params.Regul_TGV_alpha1 = 0.9; % parameter to control the first-order term
params.Regul_TGV_alpha0 = 0.3; % parameter to control the second-order term
params.Regul_Iterations = 500; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'ROFLLT') == 1)
params.Regul_Lambda_ROF_term = 1500; % regularisation parameter for ROF term
params.Regul_Lambda_LLT_term = 600; % regularisation parameter for LLT term
params.Regul_time_step = 0.00025; % time marching parameter (convergence)
params.Regul_Iterations = 3500; % inner iterations number for regularisation
elseif (strcmp(regulariser, 'Diff2nd') == 1)
params.Regul_Lambda_Diffusion = 1000; % regularisation parameter
params.Regul_FuncDiff_Type = 'Huber'; % selecting edge-preserving function
params.Regul_sigmaEdge = 0.1; % edge-preserving parameter
params.Regul_time_step = 0.005; % time marching parameter (convergence)
params.Regul_Iterations = 800; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'Diff4nd') == 1)
params.Regul_Lambda_AnisDiff4th = 20000; % regularisation parameter
params.Regul_sigmaEdge = 0.3; % edge-preserving parameter
params.Regul_time_step = 0.0007; % time marching parameter (convergence)
params.Regul_Iterations = 500; % inner iterations number for regularisation  
else
   error('Regulariser is not selected');
end

[x_result, error_vec] = ADMM_REC(params);

figure; imshow(reshape(x_result, N,N), [0 3]); title('Reconstructed image');