% A demo script to reconstruct 2D sythetic parallel-beam data which have been produced by TomoPhantom package (https://github.com/dkazanc/TomoPhantom). 
% The obtained sinogram is analytic and the inverse crime is avoided to a degree.

% The reconstruction method is the regularised Alternating Direction method of
% Multipliers (ADMM method)

% Requirements: ASTRA-toolbox with GPU-enabled device, SPOT operator, the TomoPhantom package, and the CCPi-Regularisation toolkit. 
% See ReadMe for more information and references

close all;clc;clear;
% adding paths
fsep = '/';
% TomoPhantom paths (TomoPhantom needs to be compiled first)
pathtoTomoPhantom = sprintf(['..' fsep 'supplementary' fsep 'TomoPhantom' fsep 'Wrappers' fsep 'MATLAB' fsep 'compiled' fsep], 1i);
addpath(pathtoTomoPhantom);
pathtoTomoPhantom2 = sprintf(['..' fsep 'supplementary' fsep 'TomoPhantom' fsep 'Wrappers' fsep 'MATLAB' fsep 'supplem' fsep], 1i);
addpath(pathtoTomoPhantom2);
pathtoModels = sprintf(['..' fsep 'supplementary' fsep 'TomoPhantom' fsep 'PhantomLibrary' fsep 'models' fsep 'Phantom2DLibrary.dat'], 1i);
% Regularisation Toolkit path to compiled library (CCPi-RGTtk needs to be compiled first)
pathtoRGLtk = sprintf(['..' fsep 'supplementary' fsep 'CCPi-Regularisation-Toolkit' fsep 'Wrappers' fsep 'Matlab' fsep 'mex_compile' fsep 'installed'], 1i);
addpath(pathtoRGLtk);

pathmainFunc = sprintf(['..' fsep 'main_func'], 1i);
addpath(pathmainFunc); 
pathSupp = sprintf(['..' fsep 'supplementary'], 1i);
addpath(pathSupp);
%%
fprintf('\n %s\n', 'Generating a phantom and projection data using the TomoPhantom package...');
ModelNo = 14; % Select a model from Phantom2DLibrary.dat
% Define phantom dimensions
N = 512; % x-y size (squared image)

PhantomExact = TomoP2DModel(ModelNo,N,pathtoModels); % Generate 2D phantom:

anglesNumb = round(0.5*N); % number of projection angles
anglesDegrees = linspace(0,180,anglesNumb); % projection angles
Detectors = round(sqrt(2)*N); % number of detectors

% generate an ideal analytical sinogram 
[sinoIdeal] = TomoP2DModelSino(ModelNo, N, Detectors, single(anglesDegrees), pathtoModels, 'astra'); 

% adding Poisson noise
dose =  1e4; % photon flux (controls noise level)
[sinoNoise,rawdata] = add_noise(sinoIdeal, dose, 'Poisson'); % adding Poisson noise

figure; 
subplot(1,3,1); imagesc(PhantomExact, [0 1]); daspect([1 1 1]); title('Exact phantom'); colormap hot; 
subplot(1,3,2); imshow(sinoIdeal, [ ]); title('Ideal sinogram'); colormap hot;
subplot(1,3,3); imshow(sinoNoise, [ ]); title('Noisy sinogram'); colormap hot;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Define projection geometry using ASTRA-toolbox
proj_geom = astra_create_proj_geom('parallel', 1, Detectors, anglesDegrees*pi/180);
vol_geom = astra_create_vol_geom(N,N);
% 
% % Create the Spot operator for ASTRA using the GPU.
A = opTomo('cuda', proj_geom, vol_geom);
%%
clear params;
params.A = A; % projection matrix
params.sino = sinoNoise(:); % vectorised sinogram
params.phantomExact = PhantomExact; % exact phantom
params.iterADMM = 35; % the number of outer ADMM iterations
params.rhoADMM = 50; % Convergence-related ADMM constant
params.alphaADMM = 1; % ADMM variable (relaxation of ADMM iterations)
params.iterKrylov = 30; % the number of iterations for Krylov method to solve SLAE
params.TolKrylov = 1e-5; % tolerance parameter for Krylov updates

%>>>>>>>>>>>> Regularisation block <<<<<<<<<<<<<<
params.Regul_device = 'gpu'; % select 'cpu' or 'gpu' device for regularisation

% Select preferable regulariser (see more information on CCPi-RGL toolkit).

% Note that the provided regularisation parameters bellow were optimally
% selected, however, parameters can slightly vary with respect to different 
% noise initialisations

regulariser = 'ROF_TV'; % Rudin-Osher-Fatemi Total Variation functional 
% regulariser = 'FGP_TV'; % Fast-gradient-projection Total Variation method
% regulariser = 'SB_TV'; % Split Bregman Total Variation method
% regulariser = 'TGV'; % Total Generilised Variation method
% regulariser = 'ROFLLT'; % ROF-LLT model (higher order regulariser)
% regulariser = 'Diff2nd'; % Nonlinear diffusion regulariser of the 2nd Or
% regulariser = 'Diff4nd'; % Anisotropic diffusion of the 4th order

if (strcmp(regulariser, 'ROF_TV') == 1)
params.Regul_Lambda_ROFTV = 150; % regularisation parameter
params.Regul_time_step = 0.00008; % time marching parameter (convergence)
params.Regul_Iterations = 4000; % inner iterations number for regularisation
elseif (strcmp(regulariser, 'FGP_TV') == 1)
params.Regul_Lambda_FGPTV = 150; % regularisation parameter
params.Regul_Iterations = 100; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'SB_TV') == 1)
params.Regul_Lambda_SBTV = 150; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'TGV') == 1)
params.Regul_Lambda_TGV = 100; % regularisation parameter
params.Regul_TGV_alpha1 = 0.9; % parameter to control the first-order term
params.Regul_TGV_alpha0 = 1; % parameter to control the second-order term
params.Regul_Iterations = 350; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'ROFLLT') == 1)
params.Regul_Lambda_ROF_term = 150; % regularisation parameter for the ROF term
params.Regul_Lambda_LLT_term = 50; % regularisation parameter for the LLT term
params.Regul_time_step = 0.00008; % time marching parameter (convergence)
params.Regul_Iterations = 4000; % inner iterations number for regularisation
elseif (strcmp(regulariser, 'Diff2nd') == 1)
params.Regul_Lambda_Diffusion = 250; % regularisation parameter
params.Regul_FuncDiff_Type = 'Huber'; % selecting edge-preserving function
params.Regul_sigmaEdge = 0.1; % edge-preserving parameter
params.Regul_time_step = 0.002; % time marching parameter (convergence)
params.Regul_Iterations = 500; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'Diff4nd') == 1)
params.Regul_Lambda_AnisDiff4th = 1500; % regularisation parameter
params.Regul_sigmaEdge = 0.22; % edge-preserving parameter
params.Regul_time_step = 0.0003; % time marching parameter (convergence)
params.Regul_Iterations = 600; % inner iterations number for regularisation  
else
   error('Regulariser is not selected');
end

[x_result, error_vec] = ADMM_REC(params);

figure; 
subplot(1,2,1); imshow(reshape(x_result, N,N), [0 1]); title('Reconstructed Phantom');
subplot(1,2,2); plot(error_vec);
%%