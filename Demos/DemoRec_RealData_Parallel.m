% A demo script to reconstruct 3D parallel-beam real-data (Collected at I12 branchline, Diamond Light Source).

% This is ordered-subset regularised iterative FISTA reconstriction method with PWLS data
% fidelity

% Requirements: ASTRA-toolbox with GPU-enabled device and the CCPi-Regularisation toolkit. 
% See ReadMe for more information and links

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
pathtoData = sprintf(['..' fsep 'data'], 1i);
addpath(pathtoData);
%%
fprintf('\n %s\n', 'Loading real data...');
[Sino3D,  Weights3D, anglesRad, Detectors] = exportDemoRD2Data();
N = 1050; %   Reconstruction domain dimension
SlicesNo = size(Sino3D,3); % number of z-slices

slice = 10;
figure;
subplot(1,2,1); imshow(squeeze(Sino3D(:,:,slice))', [ ]); colormap hot; title('Sinogram slice');
subplot(1,2,2); imshow(squeeze(Sino3D(:,slice,:))', [ ]); colormap hot; title('Y-Slice');
%%
fprintf('\n %s\n', 'Generating geometry using ASTRA toolbox...');

% 3D geometry to reconstruct the whole subset
% proj_geom = astra_create_proj_geom('parallel3d', 1.0, 1.0, SlicesNo, Detectors, anglesRad);
% vol_geom = astra_create_vol_geom(N,N,SlicesNo);

% 2D geometry to reconstruct one selected slice
proj_geom = astra_create_proj_geom('parallel', 1.0, Detectors, anglesRad);
vol_geom = astra_create_vol_geom(N,N);

%%
fprintf('%s\n', 'Reconstruction using FISTA-OS-PWLS with 2D regularisation...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D(:,:,slice); % sinogram
params.iterFISTA = 25; %max number of outer iterations
params.subsets = 10; % the number of subsets
params.weights = Weights3D(:,:,slice)./max(Weights3D(:,:,slice)); % statistical weights for PWLS
params.show = 1; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 2.5; 
%>>>>>>>>>>>> Regularisation block <<<<<<<<<<<<<<
params.Regul_device = 'cpu'; % select 'cpu' or 'gpu' device for regularisation
% Select preferable regulariser (see more information on CCPi-RGLTK):

regulariser = 'ROF_TV'; % Rudin-Osher-Fatemi Total Variation functional 
% regulariser = 'FGP_TV'; % Fast-gradient-projection Total Variation method
% regulariser = 'SB_TV'; % Split Bregman Total Variation method
% regulariser = 'NonlDiff'; % Nonlinear diffusion regulariser
% regulariser = 'AnisoDiff4th'; % Anisotropic diffusion of 4th order 

if (strcmp(regulariser, 'ROF_TV') == 1)
params.Regul_Lambda_ROFTV = 15000; % regularisation parameter
params.Regul_time_step = 0.005; % time marching parameter (convergence)
params.Regul_Iterations = 50; % inner iterations number for regularisation
elseif (strcmp(regulariser, 'FGP_TV') == 1)
params.Regul_Lambda_FGPTV = 1000; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'SB_TV') == 1)
params.Regul_Lambda_SBTV = 1000; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'NonlDiff') == 1)
params.Regul_Lambda_Diffusion = 2000; % regularisation parameter
params.Regul_FuncDiff_Type = 'Huber'; % selecting edge-preserving function
params.Regul_sigmaEdge = 0.001; % edge-preserving parameter
params.Regul_time_step = 0.015; % time marching parameter (convergence)
params.Regul_Iterations = 120; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'AnisoDiff4th') == 1)
params.Regul_Lambda_AnisDiff4th = 100000; % regularisation parameter
params.Regul_sigmaEdge = 0.15; % edge-preserving parameter
params.Regul_time_step = 0.0015; % time marching parameter (convergence)
params.Regul_Iterations = 150; % inner iterations number for regularisation  
else
   error('Regulariser is not selected');
end
tic; [X_FISTA, output] = FISTA_REC(params); toc;
figure; imshow(X_FISTA,[0 params.maxvalplot]); title('FISTA-PWLS-reg reconstruction'); colorbar;
%%
fprintf('%s\n', 'Reconstruction using FISTA-OS-GH with 2D regularisation...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D(:,:,slice); % sinogram
params.iterFISTA = 25; %max number of outer iterations
params.subsets = 10; % the number of subsets
params.weights = Weights3D(:,:,slice)./max(Weights3D(:,:,slice)); % statistical weights for PWLS
params.show = 1; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 2.5; 
%>>>>>>>>>>>> Group-Huber data fidelity <<<<<<<<<<<<<<
params.Ring_LambdaR_L1 = 0.02; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 20; % to boost ring removal procedure
%>>>>>>>>>>>> Regularisation block <<<<<<<<<<<<<<
params.Regul_device = 'cpu'; % select 'cpu' or 'gpu' device for regularisation
% Select preferable regulariser (see more information on CCPi-RGLTK):

regulariser = 'ROF_TV'; % Rudin-Osher-Fatemi Total Variation functional 
% regulariser = 'FGP_TV'; % Fast-gradient-projection Total Variation method
% regulariser = 'SB_TV'; % Split Bregman Total Variation method
% regulariser = 'NonlDiff'; % Nonlinear diffusion regulariser
% regulariser = 'AnisoDiff4th'; % Anisotropic diffusion of 4th order 

if (strcmp(regulariser, 'ROF_TV') == 1)
params.Regul_Lambda_ROFTV = 15000; % regularisation parameter
params.Regul_time_step = 0.005; % time marching parameter (convergence)
params.Regul_Iterations = 50; % inner iterations number for regularisation
elseif (strcmp(regulariser, 'FGP_TV') == 1)
params.Regul_Lambda_FGPTV = 1000; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'SB_TV') == 1)
params.Regul_Lambda_SBTV = 1000; % regularisation parameter
params.Regul_Iterations = 80; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'NonlDiff') == 1)
params.Regul_Lambda_Diffusion = 2000; % regularisation parameter
params.Regul_FuncDiff_Type = 'Huber'; % selecting edge-preserving function
params.Regul_sigmaEdge = 0.001; % edge-preserving parameter
params.Regul_time_step = 0.015; % time marching parameter (convergence)
params.Regul_Iterations = 120; % inner iterations number for regularisation  
elseif (strcmp(regulariser, 'AnisoDiff4th') == 1)
params.Regul_Lambda_AnisDiff4th = 100000; % regularisation parameter
params.Regul_sigmaEdge = 0.15; % edge-preserving parameter
params.Regul_time_step = 0.0015; % time marching parameter (convergence)
params.Regul_Iterations = 150; % inner iterations number for regularisation  
else
   error('Regulariser is not selected');
end
tic; [X_FISTA_GH, outputGH] = FISTA_REC(params); toc;
figure; imshow(X_FISTA_GH,[0 params.maxvalplot]); title('FISTA-GH-reg reconstruction'); colorbar;
%%
% fprintf('%s\n', 'Reconstruction using FISTA-OS-Studentt with 2D regularisation...');
% clear params
% % define parameters
% params.proj_geom = proj_geom; % pass geometry to the function
% params.vol_geom = vol_geom;
% params.sino = Sino3D(:,:,slice); % sinogram
% params.iterFISTA = 150; %max number of outer iterations
% % params.subsets = 10; % the number of subsets
% params.weights = Weights3D(:,:,slice)./max(Weights3D(:,:,slice)); % statistical weights for PWLS
% params.show = 1; % visualize reconstruction on each iteration
% params.slice = 1; params.maxvalplot = 2.5; 
% %>>>>>>>>>>>> Students't data fidelity <<<<<<<<<<<<<<
% params.fidelity = 'students_data';
% params.L_const = 500; % accelerate covergence with Students't
% 
% %>>>>>>>>>>>> Regularisation block <<<<<<<<<<<<<<
% params.Regul_device = 'gpu'; % select 'cpu' or 'gpu' device for regularisation
% % Select preferable regulariser (see more information on CCPi-RGLTK):
% 
% regulariser = 'ROF_TV'; % Rudin-Osher-Fatemi Total Variation functional 
% % regulariser = 'FGP_TV'; % Fast-gradient-projection Total Variation method
% % regulariser = 'SB_TV'; % Split Bregman Total Variation method
% % regulariser = 'NonlDiff'; % Nonlinear diffusion regulariser
% % regulariser = 'AnisoDiff4th'; % Anisotropic diffusion of 4th order 
% 
% if (strcmp(regulariser, 'ROF_TV') == 1)
% params.Regul_Lambda_ROFTV = 8; % regularisation parameter
% params.Regul_time_step = 0.005; % time marching parameter (convergence)
% params.Regul_Iterations = 50; % inner iterations number for regularisation
% elseif (strcmp(regulariser, 'FGP_TV') == 1)
% params.Regul_Lambda_FGPTV = 1000; % regularisation parameter
% params.Regul_Iterations = 80; % inner iterations number for regularisation  
% elseif (strcmp(regulariser, 'SB_TV') == 1)
% params.Regul_Lambda_SBTV = 1000; % regularisation parameter
% params.Regul_Iterations = 80; % inner iterations number for regularisation  
% elseif (strcmp(regulariser, 'NonlDiff') == 1)
% params.Regul_Lambda_Diffusion = 2000; % regularisation parameter
% params.Regul_FuncDiff_Type = 'Huber'; % selecting edge-preserving function
% params.Regul_sigmaEdge = 0.001; % edge-preserving parameter
% params.Regul_time_step = 0.015; % time marching parameter (convergence)
% params.Regul_Iterations = 120; % inner iterations number for regularisation  
% elseif (strcmp(regulariser, 'AnisoDiff4th') == 1)
% params.Regul_Lambda_AnisDiff4th = 100000; % regularisation parameter
% params.Regul_sigmaEdge = 0.15; % edge-preserving parameter
% params.Regul_time_step = 0.0015; % time marching parameter (convergence)
% params.Regul_Iterations = 150; % inner iterations number for regularisation  
% else
%    error('Regulariser is not selected');
% end
% tic; [X_FISTA_Stud, outputStud] = FISTA_REC(params); toc;
% figure; imshow(X_FISTA_Stud,[0 params.maxvalplot]); title('FISTA-Studentt-reg reconstruction'); colorbar;
