% A demo script to reconstruct 2D sythetic parallel-beam data which have been produced by TomoPhantom package. 
% The sinogram is analytic and inverse crime is avoided to a degree.

% This is ordered-subset regularised iterative FISTA reconstriction method with
% Group-Huber and Students't data fidelities 
% to reconstruct erroneous data with artifacts - zingers and stripes (rings)

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

% Artifacts 
[sino_zingers] = add_zingers(sinoIdeal, 0.5, 10); % adding zingers
[sino_zingers_stripes] = add_stripes(sino_zingers, 1, 1); % adding stripes

% adding Poisson noise
dose =  1e4; % photon flux (controls noise level)
[sinoNoise,rawdata] = add_noise(sino_zingers_stripes, dose, 'Poisson'); % adding Poisson noise

figure; 
subplot(1,3,1); imagesc(PhantomExact, [0 1]); daspect([1 1 1]); title('Exact phantom'); colormap hot; 
subplot(1,3,2); imshow(sinoIdeal, [ ]); title('Ideal sinogram'); colormap hot;
subplot(1,3,3); imshow(sinoNoise, [ ]); title('Noisy sinogram with artifacts'); colormap hot;
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
fprintf('%s\n', 'Reconstruction using FISTA-OS-PWLS without regularisation...');
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
fprintf('\n %s\n', 'Reconstruction using FISTA-OS-PWLS-GH with regularisation...');
clear params regulariser
params.proj_geom = proj_geom; % pass ASTRA geometry 
params.vol_geom = vol_geom; % pass ASTRA geometry 
params.sino = sinoNoise'; % sinogram
params.iterFISTA = 30; % max number of FISTA iterations
params.subsets = 12; % the number of subsets
params.phantomExact = PhantomExact; % ideal phantom
params.weights = rawdata'./max(rawdata(:)); % normalised raw data as a weight for PWLS
params.show = 1; % visualise reconstruction on each iteration
params.maxvalplot = 1; % max intensity of recovered image
%>>>>>>>>>>>> Group-Huber data fidelity <<<<<<<<<<<<<<
params.Ring_LambdaR_L1 = 0.02; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 20; % to boost ring removal procedure

%>>>>>>>>>>>> Regularisation block <<<<<<<<<<<<<<
params.Regul_device = 'cpu'; % select 'cpu' or 'gpu' device for regularisation
% Select preferable regulariser (see more information on CCPi-RGLTK):

% regulariser = 'ROF_TV'; % Rudin-Osher-Fatemi Total Variation functional 
 regulariser = 'FGP_TV'; % Fast-gradient-projection Total Variation method
% regulariser = 'SB_TV'; % Split Bregman Total Variation method
% regulariser = 'NonlDiff'; % Nonlinear diffusion regulariser
% regulariser = 'AnisoDiff4th'; % Anisotropic diffusion of 4th order 

if (strcmp(regulariser, 'ROF_TV') == 1)
params.Regul_Lambda_ROFTV = 1000; % regularisation parameter
params.Regul_time_step = 0.005; % time marching parameter (convergence)
params.Regul_Iterations = 50; % inner iterations number for regularisation
elseif (strcmp(regulariser, 'FGP_TV') == 1)
params.Regul_Lambda_FGPTV = 500; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'SB_TV') == 1)
params.Regul_Lambda_SBTV = 500; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'NonlDiff') == 1)
params.Regul_Lambda_Diffusion = 100; % regularisation parameter
params.Regul_FuncDiff_Type = 'Huber'; % selecting edge-preserving function
params.Regul_sigmaEdge = 0.001; % edge-preserving parameter
params.Regul_time_step = 0.015; % time marching parameter (convergence)
params.Regul_Iterations = 120; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'AnisoDiff4th') == 1)
params.Regul_Lambda_AnisDiff4th = 10000; % regularisation parameter
params.Regul_sigmaEdge = 0.09; % edge-preserving parameter
params.Regul_time_step = 0.0015; % time marching parameter (convergence)
params.Regul_Iterations = 150; % inner iterations number for regularisation  
else
   error('Regulariser is not selected');
end
%>>>>>>>>>>>>>>>>>>>> end <<<<<<<<<<<<<<<<<<<<<<<
tic; [X_FISTA_regul, output] = FISTA_REC(params); toc; 

error_FISTA_reg = output.Resid_error; obj_FISTA_reg = output.objective;
fprintf('%s %.4f\n', 'Min RMSE for regularised FISTA-OS-PWLS-GH reconstruction is:', min(error_FISTA_reg(:)));

Resid = (PhantomExact - X_FISTA_regul).^2;
figure(5);
subplot(1,2,1); imshow(X_FISTA_regul,[0 params.maxvalplot]); title('FISTA-PWLS-GH reconstruction'); colorbar;
subplot(1,2,2); imshow(Resid,[0 0.1]);  title('residual'); colorbar;
figure(6);
subplot(1,2,1); plot(error_FISTA_reg);  title('RMSE plot'); 
subplot(1,2,2); plot(obj_FISTA_reg);  title('Objective plot'); 
%%
fprintf('\n %s\n', 'Reconstruction using FISTA-OS-Students-t with regularisation...');
clear params regulariser
params.proj_geom = proj_geom; % pass ASTRA geometry 
params.vol_geom = vol_geom; % pass ASTRA geometry 
params.sino = sinoNoise'; % sinogram
params.iterFISTA = 40; % max number of FISTA iterations
params.subsets = 16; % the number of subsets
params.phantomExact = PhantomExact; % ideal phantom
params.weights = rawdata'./max(rawdata(:)); % normalised raw data as a weight for PWLS
params.show = 1; % visualise reconstruction on each iteration
params.maxvalplot = 1; % max intensity of recovered image
%>>>>>>>>>>>> Students't data fidelity <<<<<<<<<<<<<<
params.fidelity = 'students_data';
params.L_const = 3500; % accelerate covergence with Students't

%>>>>>>>>>>>> Regularisation block <<<<<<<<<<<<<<
params.Regul_device = 'gpu'; % select 'cpu' or 'gpu' device for regularisation
% Select preferable regulariser (see more information on CCPi-RGLTK):

% regulariser = 'ROF_TV'; % Rudin-Osher-Fatemi Total Variation functional 
regulariser = 'FGP_TV'; % Fast-gradient-projection Total Variation method
% regulariser = 'SB_TV'; % Split Bregman Total Variation method
% regulariser = 'NonlDiff'; % Nonlinear diffusion regulariser
% regulariser = 'AnisoDiff4th'; % Anisotropic diffusion of 4th order 

if (strcmp(regulariser, 'ROF_TV') == 1)
params.Regul_Lambda_ROFTV = 50; % regularisation parameter
params.Regul_time_step = 0.005; % time marching parameter (convergence)
params.Regul_Iterations = 50; % inner iterations number for regularisation
elseif (strcmp(regulariser, 'FGP_TV') == 1)
params.Regul_Lambda_FGPTV = 25; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'SB_TV') == 1)
params.Regul_Lambda_SBTV = 25; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'NonlDiff') == 1)
params.Regul_Lambda_Diffusion = 20; % regularisation parameter
params.Regul_FuncDiff_Type = 'Huber'; % selecting edge-preserving function
params.Regul_sigmaEdge = 0.001; % edge-preserving parameter
params.Regul_time_step = 0.015; % time marching parameter (convergence)
params.Regul_Iterations = 120; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'AnisoDiff4th') == 1)
params.Regul_Lambda_AnisDiff4th = 1000; % regularisation parameter
params.Regul_sigmaEdge = 0.09; % edge-preserving parameter
params.Regul_time_step = 0.0015; % time marching parameter (convergence)
params.Regul_Iterations = 150; % inner iterations number for regularisation  
else
   error('Regulariser is not selected');
end
%>>>>>>>>>>>>>>>>>>>> end <<<<<<<<<<<<<<<<<<<<<<<
tic; [X_FISTA_regul, output] = FISTA_REC(params); toc; 

error_FISTA_reg = output.Resid_error; obj_FISTA_reg = output.objective;
fprintf('%s %.4f\n', 'Min RMSE for regularised FISTA-OS-Students-t reconstruction is:', min(error_FISTA_reg(:)));

Resid = (PhantomExact - X_FISTA_regul).^2;
figure(5);
subplot(1,2,1); imshow(X_FISTA_regul,[0 params.maxvalplot]); title('FISTA-OS-Students-t reconstruction'); colorbar;
subplot(1,2,2); imshow(Resid,[0 0.1]);  title('residual'); colorbar;
figure(6);
subplot(1,2,1); plot(error_FISTA_reg);  title('RMSE plot'); 
subplot(1,2,2); plot(obj_FISTA_reg);  title('Objective plot'); 
%%