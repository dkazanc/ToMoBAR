% A demo script to generate and reconstruct 3D sythetic cone-beam data.

% This is regularised iterative FISTA ordered-subset reconstriction method with PWLS data
% fidelity

% Requirements: ASTRA-toolbox with GPU-enabled device, the TomoPhantom package, the CCPi-Regularisation toolkit. 
% See ReadMe for more information and links

close all;clc;clear;
adding_paths % ading all required paths (modify if required)
%%
% Generate 3D phantom:
fprintf('\n %s\n', 'Generating a phantom using the TomoPhantom package...');
ModelNo = 11; % Select a model from Phantom3DLibrary.dat
% Define phantom dimensions
N = 256; % x-y-z size

PhantomExact = TomoP3DModel(ModelNo,N,pathtoModels); % Generate 3D phantom

anglesNumb = round(0.5*N); % number of projection angles
anglesRad = linspace(0,2*pi,anglesNumb); % projection angles
Detectors = round(sqrt(2)*N); % number of detectors

% check 3 views
figure; 
slice = round(0.5*N);
subplot(1,3,1); imagesc(PhantomExact(:,:,slice), [0 1]); daspect([1 1 1]); colormap hot; title('Axial Slice');
subplot(1,3,2); imagesc(squeeze(PhantomExact(:,slice,:)), [0 1]); daspect([1 1 1]); colormap hot; title('Y-Slice');
subplot(1,3,3); imagesc(squeeze(PhantomExact(slice,:,:)), [0 1]); daspect([1 1 1]); colormap hot; title('X-Slice');
%%
fprintf('\n %s\n', 'Generating cobe-beam projection data using ASTRA toolbox...');
source_origin = 7*N;
origin_det = 1.25*N;

proj_geom = astra_create_proj_geom('cone', 1.0, 1.0, N, Detectors, anglesRad, source_origin, origin_det);
vol_geom = astra_create_vol_geom(N,N,N);


[sino_id, ProjData3D] = astra_create_sino3d_cuda(PhantomExact, proj_geom, vol_geom);
astra_mex_data3d('delete', sino_id);
ProjData3D = single(ProjData3D);

fprintf('\n %s\n', 'Adding Poisson noise...'); 
dose =  5e4; % photon flux (controls noise level)
[ProjData_noise3D,ProjData_raw] = add_noise(ProjData3D, dose, 'Poisson'); % adding Poisson noise

figure; 
slice = round(0.5*N);
subplot(1,2,1); imshow(squeeze(ProjData_noise3D(:,:,slice))', [ ]); colormap hot; title('Sinogram slice');
subplot(1,2,2); imshow(squeeze(ProjData_noise3D(:,slice,:))', [ ]); colormap hot; title('Y-Slice');
%
%%
fprintf('%s\n', 'Reconstruction using FISTA-OS-PWLS with 3D regularisation...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = ProjData_noise3D; % sinogram
params.iterFISTA = 15; %max number of outer iterations
params.phantomExact = PhantomExact; % ideal phantom
params.weights = ProjData_raw./max(ProjData_raw(:)); % statistical weights for PWLS
params.subsets = 10; % the number of subsets
params.show = 1; % visualize reconstruction on each iteration
params.slice = 128; params.maxvalplot = 1.1; 
%>>>>>>>>>>>> Regularisation block <<<<<<<<<<<<<<
params.Regul_device = 'cpu'; % select 'cpu' or 'gpu' device for regularisation
params.Regul_Dimension = '3D'; % select between 2D or 3D regularisation
% Select preferable regulariser (see more information on CCPi-RGLTK):

regulariser = 'ROF_TV'; % Rudin-Osher-Fatemi Total Variation functional 
% regulariser = 'FGP_TV'; % Fast-gradient-projection Total Variation method
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

tic; [X_FISTA, output] = FISTA_REC(params); toc;

error_FISTA = output.Resid_error; obj_FISTA = output.objective;
fprintf('%s %.4f\n', 'Min RMSE for FISTA-PWLS reconstruction is:', min(error_FISTA(:)));

Resid3D = (PhantomExact - X_FISTA).^2;
figure(2);
subplot(1,2,1); imshow(X_FISTA(:,:,params.slice),[0 params.maxvalplot]); title('FISTA-LS reconstruction'); colorbar;
subplot(1,2,2); imshow(Resid3D(:,:,params.slice),[0 0.1]);  title('residual'); colorbar;
figure(3);
subplot(1,2,1); plot(error_FISTA);  title('RMSE plot'); 
subplot(1,2,2); plot(obj_FISTA);  title('Objective plot'); 
%%
