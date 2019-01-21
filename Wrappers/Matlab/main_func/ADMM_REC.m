function [x, Resid_error] = ADMM_REC(params)

% <<<<<<<<< Regularised ADMM method [1-4] using ASTRA-Toolbox >>>>>>>>>>>

% Dependencies: ASTRA-toolbox, SPOT operator, CCPi-Regularisation Toolkit

% ___Input___:
% params.[] file:
%       - .A (projection matrix formed by SPOT operator) [required]
%       - .sino (vectorised 2D sinogram) [required]
%       - .phantomExact (ideal phantom if available)
%       - .iterADMM (outer ADMM iterations, default 30)
%       - .rhoADMM (augmented Lagrangian parameter, default 1)
%       - .alphaADMM (over-relaxation parameter, default 1)
%       - .iterKrylov (inner Krylov iterations, default 25)
%       - .TolKrylov (tolerance for inner Krylov iterations, default 1e-4)

%-------------Regularisation (main parameters)------------------
%       - .Regul_device (select 'cpu' or 'gpu' device, 'cpu' is default)
%       - .Regul_tol (tolerance to terminate regul iterations, default 1.0e-05)
%       - .Regul_Iterations (iterations for the selected penalty, default 25)
%       - .Regul_time_step (some penalties require time marching parameter, default 0.01)
%       - .Regul_Dimension ('2D' or '3D' way to apply regularisation, '2D' is the default)
%-------------Regularisation choices------------------
%       1 .Regul_Lambda_ROFTV (ROF-TV regularisation parameter)
%       2 .Regul_Lambda_FGPTV (FGP-TV regularisation parameter)
%       3 .Regul_Lambda_SBTV (SplitBregman-TV regularisation parameter)
%       4 .Regul_Lambda_TGV (TGV regularisation parameter)
%       5 .Regul_Lambda_ROF_term (ROF related term of ROF-LLT higher order regularisation scheme)
%       6 .Regul_Lambda_LLT_term (LLT related term of ROF-LLT higher order regularisation scheme)
%       7 .Regul_Lambda_Diffusion (Nonlinear diffusion regularisation parameter)
%       8 .Regul_Lambda_AnisDiff4th (Anisotropic diffusion of higher order regularisation parameter)
%       ... more to be added, see CCPi-RegularisationToolkit for updates and info

% ___Output___:
% x - Reconstructed image (vectorised)
% error_vec - vector of RMSE errors

% References:
% 1. Distributed Optimization and Statistical Learning via the Alternating
% Direction Method of Multipliers, S. Boyd et al. 2011, IEEE TMI
% 2. A Splitting-Based Iterative Algorithm for Accelerated Statistical
% X-Ray CT Reconstruction, S. Ramani and J. Fessler, 2012
% 3. Plug-and-Play priors for Model Based reconstruction,  S. V. Venkatakrishnan, 2013
% 4. Plug-and-Play ADMM for image restoration: Fixed-point convergence and
% applications, S. H. Chan, 2017, IEEE TCI

% <License>
% GPLv3 license (ASTRA toolbox derrivative)

if (isfield(params,'A'))
    A = params.A;
    n_vox = size(A,2); % reconstructed object dim - N x N
    %m_dim = size(A,1); % data dimensions - M x P
    N = sqrt(n_vox); % squared Dim of image
else
    error('%s \n', 'Please provide the projection matrix A');
end
if (isfield(params,'sino'))
    b = params.sino;
else
    error('%s \n', 'Please provide a sinogram');
end
if (isfield(params,'phantomExact'))
    phantomExact = params.phantomExact;
else
    phantomExact = 'none';
end
if (isfield(params,'iterADMM'))
    iterADMM = params.iterADMM;
else
    iterADMM = 25;
end
if (isfield(params,'rhoADMM'))
    rho_const = params.rhoADMM;
else
    rho_const = 1;
end
if (isfield(params,'alphaADMM'))
    alpha = params.alphaADMM;
else
    alpha = 1;
end
if (isfield(params,'iterKrylov'))
    iterKrylov = params.iterKrylov;
else
    iterKrylov = 30;
end
if (isfield(params,'TolKrylov'))
    TolKrylov = params.TolKrylov;
else
    TolKrylov = 1.0e-04;
end
% if (isfield(params,'objective'))
%     objective_calc = params.objective;
% else
%     objective_calc = 0;
% end

% <<<<<<<<<<<<< Regularisation settings >>>>>>>>>>>>>>
device = 0; % (cpu)
if (isfield(params,'Regul_device'))
    if (strcmp(params.Regul_device, 'gpu') == 1)
        device = 1; % (gpu)
    end
end
if (isfield(params,'Regul_tol'))
    tol = params.Regul_tol;
else
    tol = 1.0e-05;
end
if (isfield(params,'Regul_Iterations'))
    IterationsRegul = params.Regul_Iterations;
else
    IterationsRegul = 25;
end
if (isfield(params,'Regul_time_step'))
    Regul_time_step = params.Regul_time_step;
else
    Regul_time_step = 0.01;
end
if (isfield(params,'Regul_sigmaEdge'))
    sigmaEdge = params.Regul_sigmaEdge;
else
    sigmaEdge = 0.01; % edge-preserving parameter
end
if (isfield(params,'Regul_Lambda_ROFTV'))
    lambdaROF_TV = params.Regul_Lambda_ROFTV;
    fprintf('\n %s\n', 'ROF-TV regularisation is enabled...');
else
    lambdaROF_TV = 0;
end
if (isfield(params,'Regul_Lambda_FGPTV'))
    lambdaFGP_TV = params.Regul_Lambda_FGPTV;
    fprintf('\n %s\n', 'FGP-TV regularisation is enabled...');
else
    lambdaFGP_TV = 0;
end
if (isfield(params,'Regul_Lambda_SBTV'))
    lambdaSB_TV = params.Regul_Lambda_SBTV;
    fprintf('\n %s\n', 'SB-TV regularisation is enabled...');
else
    lambdaSB_TV = 0;
end
if (isfield(params,'Regul_Lambda_TGV'))
    lambdaTGV = params.Regul_Lambda_TGV;
    
    if (isfield(params,'Regul_TGV_alpha0'))
        alpha0 = params.Regul_TGV_alpha0;
    else
        alpha0 = 1;
    end
    if (isfield(params,'Regul_TGV_alpha1'))
        alpha1 = params.Regul_TGV_alpha1;
    else
        alpha1 = 0.5;
    end    
    fprintf('\n %s\n', 'Total Generilised Variation (TGV) regularisation is enabled...');
else
    lambdaTGV = 0;
end
if (isfield(params,'Regul_Lambda_ROF_term'))
    lambdaROF = params.Regul_Lambda_ROF_term;
    lambdaLLT = params.Regul_Lambda_LLT_term;
    fprintf('\n %s\n', 'ROF-LLT regularisation is enabled...');
else
    lambdaROF = 0;
    lambdaLLT = 0;
end
if (isfield(params,'Regul_Lambda_Diffusion'))
    lambdaDiffusion = params.Regul_Lambda_Diffusion;
    fprintf('\n %s\n', 'Nonlinear diffusion regularisation is enabled...');
else
    lambdaDiffusion = 0;
end
if (isfield(params,'Regul_FuncDiff_Type'))
    FuncDiff_Type = params.Regul_FuncDiff_Type;
    if ((strcmp(FuncDiff_Type, 'Huber') ~= 1) && (strcmp(FuncDiff_Type, 'PM') ~= 1) && (strcmp(FuncDiff_Type, 'Tukey') ~= 1))
        error('Please select appropriate FuncDiff_Type - Huber, PM or Tukey')
    end
end
if (isfield(params,'Regul_Lambda_AnisDiff4th'))
    lambdaDiffusion4th = params.Regul_Lambda_AnisDiff4th;
    fprintf('\n %s\n', 'Regularisation with anisotropic diffusion of 4th order is enabled...');
else
    lambdaDiffusion4th = 0;
end
% <<<<<<<<<<<<<<<End of Regularisers>>>>>>>>>>>>>>>

% --------------------------------------------------
% <<<<<<<<<<<<<<< ADMM algorithm >>>>>>>>>>>>>>>>>>>
% --------------------------------------------------

x = zeros(n_vox,1);
z = zeros(n_vox,1);
u = zeros(n_vox,1);
Resid_error = zeros(iterADMM,1);

for k = 1:iterADMM
    
    rhs = A'*b + rho_const.*(z-u); % forming the rhs of the system
    x  = krylov(rhs); % passing to the linear solver
    
    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    
    % Solving inner proximal problem (weighted denoising)   
    %--------------Regularisation part (CCPi-RGLTK)---------------%
    if (lambdaROF_TV > 0)
        % ROF-TV regularisation is enabled
        if (device == 0)
            % CPU
            z = ROF_TV(reshape(single(x_hat + u), N,N), lambdaROF_TV/rho_const, IterationsRegul, Regul_time_step);
        else
            % GPU
            z = ROF_TV_GPU(reshape(single(x_hat + u), N,N), lambdaROF_TV/rho_const, IterationsRegul, Regul_time_step);
        end
    end
    if (lambdaFGP_TV > 0)
        % FGP-TV regularisation is enabled
        if (device == 0)
            % CPU
            z = FGP_TV(reshape(single(x_hat + u), N,N), lambdaFGP_TV/rho_const, IterationsRegul, tol);
        else
            % GPU
            z = FGP_TV_GPU(reshape(single(x_hat + u), N,N), lambdaFGP_TV/rho_const, IterationsRegul, tol);
        end
    end
    if (lambdaSB_TV > 0)
        % Split Bregman regularisation is enabled
        if (device == 0)
            % CPU
            z = SB_TV(reshape(single(x_hat + u), N,N), lambdaSB_TV/rho_const, IterationsRegul, tol);
        else
            % GPU
            z = SB_TV_GPU(reshape(single(x_hat + u), N,N), lambdaSB_TV/rho_const, IterationsRegul, tol);
        end
    end
    if (lambdaTGV > 0)
        % TGV regularisation is enabled
        if (device == 0)
            % CPU
             z = TGV(reshape(single(x_hat + u), N,N), lambdaTGV/rho_const, alpha1, alpha0, IterationsRegul);
        else
            % GPU
            z = TGV_GPU(reshape(single(x_hat + u), N,N), lambdaTGV/rho_const, alpha1, alpha0, IterationsRegul);
        end
    end   
    if (lambdaROF > 0)
        % LLT-ROF regularisation is enabled
        if (device == 0)
            % CPU             
            z = LLT_ROF(reshape(single(x_hat + u), N,N), lambdaROF/rho_const, lambdaLLT/rho_const, IterationsRegul, Regul_time_step); 
        else
            % GPU
            z = LLT_ROF_GPU(reshape(single(x_hat + u), N,N), lambdaROF/rho_const, lambdaLLT/rho_const, IterationsRegul, Regul_time_step); 
        end
    end  
    if (lambdaDiffusion > 0)
        % Nonlinear diffusion regularisation is enabled
        if (device == 0)
            % CPU
            z = NonlDiff(reshape(single(x_hat + u), N,N), lambdaDiffusion/rho_const, sigmaEdge, IterationsRegul, Regul_time_step, FuncDiff_Type);
        else
            % GPU
            z = NonlDiff_GPU(reshape(single(x_hat + u), N,N), lambdaDiffusion/rho_const, sigmaEdge, IterationsRegul, Regul_time_step, FuncDiff_Type);
        end
    end
    if (lambdaDiffusion4th > 0)
        % Anisotropic diffusion of 4th order regularisation is enabled
        if (device == 0)
            % CPU
            z = Diffusion_4thO(reshape(single(x_hat + u), N,N), lambdaDiffusion4th/rho_const, sigmaEdge, IterationsRegul, Regul_time_step);
        else
            % GPU
            z = Diffusion_4thO_GPU(reshape(single(x_hat + u), N,N), lambdaDiffusion4th/rho_const, sigmaEdge, IterationsRegul, Regul_time_step);
        end
    end
    z = reshape(z,n_vox, 1); % convert back to a vector
    
    u = u + (x_hat - z); % update u variable
    
    if (strcmp(phantomExact, 'none' ) == 0)
        Resid_error(k) = RMSE(x(:), phantomExact(:));
        fprintf('%s %i %s %s %.4f \n', 'Iteration Number:', k, '|', 'Error RMSE:', Resid_error(k));
    else
        fprintf('%s %i \n', 'Iteration Number:', k);
    end
    figure(10); imshow(reshape(x, N,N), [0 3]); title('Reconstructed Phantom');
end

%  Krylov solver to solve the linear system
    function dx = krylov(r)
        dx = gmres (@jtjx, r, 30, TolKrylov, iterKrylov);
        %         dx = bicgstab(@jtjx, r, TolKrylov, iterKrylov);
    end

% Callback function for matrix-vector product (called by krylov)
    function b = jtjx(sol)
        
        % Data fidelity part or lhs of equation
        data_upd = A*sol;
        x_temp = A'*(data_upd);
        
        b  = x_temp + rho_const.*sol;
    end
end
