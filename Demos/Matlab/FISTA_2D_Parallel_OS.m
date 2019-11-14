% A demo script to reconstruct 2D sythetic parallel-beam data which have been produced by TomoPhantom package. 
% The sinogram is analytic and inverse crime is avoided to a degree.

% This is ordered-subset realisation of regularised iterative FISTA reconstriction method with PWLS data
% fidelity

% Requirements: ASTRA-toolbox with GPU-enabled device, the TomoPhantom package, the CCPi-Regularisation toolkit. 
% See ReadMe for more information and links

close all;clc;clear;
adding_paths % ading all required paths (modify if required)
%%
fprintf('\n %s\n', 'Generating a phantom and projection data using the TomoPhantom package...');
ModelNo = 4; % Select a model from Phantom2DLibrary.dat
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
% using ASTRA-toolbox to set the projection geometry (parallel beam, GPU projector)
proj_geom = astra_create_proj_geom('parallel', 1, Detectors, anglesDegrees*pi/180);
vol_geom = astra_create_vol_geom(N,N);
%%
fprintf('\n %s\n', 'Reconstructing using FBP (astra-toolbox)...');
FBP = rec2Dastra(sinoNoise, (anglesDegrees*pi/180), Detectors, N, 'cpu');
figure; imagesc(FBP, [0 1]); daspect([1 1 1]); title('FBP reconstruction'); colormap hot; 
%% 
fprintf('%s\n', 'Reconstruction using FISTA-PWLS without regularisation...');
clear params
params.proj_geom = proj_geom; % pass ASTRA geometry 
params.vol_geom = vol_geom; % pass ASTRA geometry 
params.sino = sinoNoise'; % sinogram
params.iterFISTA = 15; % max number of FISTA iterations
params.subsets = 12; % the number of subsets
params.phantomExact = PhantomExact; % ideal phantom
params.weights = rawdata'./max(rawdata(:)); % normalised raw data as a weight for PWLS
params.show = 1; % visualise reconstruction on each iteration
params.maxvalplot = 1; % max intensity of recovered image
tic; [X_FISTA, output] = FISTA_REC(params); toc; 

error_FISTA = output.Resid_error; obj_FISTA = output.objective;
fprintf('%s %.4f\n', 'Min RMSE for FISTA-PWLS reconstruction is:', min(error_FISTA(:)));

Resid = (PhantomExact - X_FISTA).^2;
figure(3);
subplot(1,2,1); imshow(X_FISTA,[0 params.maxvalplot]); title('FISTA-PWLS reconstruction'); colorbar;
subplot(1,2,2); imshow(Resid,[0 0.1]);  title('residual'); colorbar;
figure(4);
subplot(1,2,1); plot(error_FISTA);  title('RMSE plot'); 
subplot(1,2,2); plot(obj_FISTA);  title('Objective plot'); 
%% 
fprintf('\n %s\n', 'Reconstruction using FISTA-PWLS with regularisation...');
clear params regulariser
params.proj_geom = proj_geom; % pass ASTRA geometry 
params.vol_geom = vol_geom; % pass ASTRA geometry 
params.sino = sinoNoise'; % sinogram
params.iterFISTA = 25; % max number of FISTA iterations
params.subsets = 10; % the number of subsets
params.phantomExact = PhantomExact; % ideal phantom
params.weights = rawdata'./max(rawdata(:)); % normalised raw data as a weight for PWLS
params.show = 1; % visualise reconstruction on each iteration
params.maxvalplot = 1; % max intensity of recovered image
%>>>>>>>>>>>> Regularisation block <<<<<<<<<<<<<<
params.Regul_device = 'cpu'; % select 'cpu' or 'gpu' device for regularisation
% Select preferable regulariser (see more information on CCPi-RGLTK):

 regulariser = 'ROF_TV'; % Rudin-Osher-Fatemi Total Variation functional 
% regulariser = 'FGP_TV'; % Fast-gradient-projection Total Variation method
% regulariser = 'SB_TV'; % Split Bregman Total Variation method
% regulariser = 'NonlDiff'; % Nonlinear diffusion regulariser
% regulariser = 'AnisoDiff4th'; % Anisotropic diffusion of 4th order 

if (strcmp(regulariser, 'ROF_TV') == 1)
params.Regul_Lambda_ROFTV = 1500; % regularisation parameter
params.Regul_time_step = 0.005; % time marching parameter (convergence)
params.Regul_Iterations = 70; % inner iterations number for regularisation
elseif (strcmp(regulariser, 'FGP_TV') == 1)
params.Regul_Lambda_FGPTV = 1000; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'SB_TV') == 1)
params.Regul_Lambda_SBTV = 1000; % regularisation parameter
params.Regul_Iterations = 40; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'NonlDiff') == 1)
params.Regul_Lambda_Diffusion = 500; % regularisation parameter
params.Regul_FuncDiff_Type = 'Huber'; % selecting edge-preserving function
params.Regul_sigmaEdge = 0.002; % edge-preserving parameter
params.Regul_time_step = 0.015; % time marching parameter (convergence)
params.Regul_Iterations = 120; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'AnisoDiff4th') == 1)
params.Regul_Lambda_AnisDiff4th = 20000; % regularisation parameter
params.Regul_sigmaEdge = 0.12; % edge-preserving parameter
params.Regul_time_step = 0.0015; % time marching parameter (convergence)
params.Regul_Iterations = 150; % inner iterations number for regularisation  
else
   error('Regulariser is not selected');
end
%>>>>>>>>>>>>>>>>>>>> end <<<<<<<<<<<<<<<<<<<<<<<
tic; [X_FISTA_regul, output] = FISTA_REC(params); toc; 

error_FISTA_reg = output.Resid_error; obj_FISTA_reg = output.objective;
fprintf('%s %.4f\n', 'Min RMSE for regularised FISTA-PWLS reconstruction is:', min(error_FISTA_reg(:)));

Resid = (PhantomExact - X_FISTA_regul).^2;
figure(5);
subplot(1,2,1); imshow(X_FISTA_regul,[0 params.maxvalplot]); title('FISTA-PWLS reconstruction'); colorbar;
subplot(1,2,2); imshow(Resid,[0 0.1]);  title('residual'); colorbar;
figure(6);
subplot(1,2,1); plot(error_FISTA_reg);  title('RMSE plot'); 
subplot(1,2,2); plot(obj_FISTA_reg);  title('Objective plot'); 
%%