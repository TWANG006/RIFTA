function [...
    X_B, Y_B, ...BRF Coordinates
    B, ... BRF
    Z_removal, Z_residual, ... full aperture results [m]
    X_P, Y_P, ...dwell grid
    T_P, ... dwell time on the dwell grid [s]
    X_dw, Y_dw, ... dwll grid coordinates [m]
    dw_range, ... dwell grid coordinates [m]
    Z_to_remove_dw, Z_removal_dw, Z_residual_dw,...dwell grid results [m]
    X_ca, Y_ca, ... clear aperture coordinates [m]
    Z_to_remove_ca, Z_removal_ca, Z_residual_ca ... clear aperture results [m]
    ] = rifta_height...
    ( X, Y...full aperture coordinates [m]
    , Z_to_remove... height to remove [m]
    , BRF_params... BRF parameters
    , X_BRF, Y_BRF...
    , Z_BRF ... averaged z & its coords
    , ca_range... Clear aperture range [pixel]
    , options... optgions for the algorithm
    )
% This function implements the height-based RIFTA where the dwell time is
% calculated as t * r = z

%% 0. Deal with default input arguments
defaultOptions = struct(...
    'Algorithm', 'Iterative-FFT',...
    'maxIters', 10,...
    'RMS_dif', 0.001e-9,...[m]
    'dwellTime_dif', 60,...[s]
    'brfMode', 'avg',...
    'samplingInterval', 1e-3,...
    'isDownSampling', false,...
    'ratio', 1,...
    'viewFullAperture', true...
    );

if nargin == 8
    options = defaultOptions;
end

%% 1. Construct the BRF using the BRF parameters
pixel_m = median(diff(X(1, :)));

brf_r = 0.5 * BRF_params.d;
[X_B, Y_B] = meshgrid(-brf_r:pixel_m:brf_r, -brf_r:pixel_m:brf_r);
Y_B = -Y_B;

if strcmpi(options.brfMode, 'avg')
    B = interp2(X_BRF, Y_BRF, Z_BRF, X_B, Y_B, 'spline'); % resize BRF
else
    A = BRF_params.A;   % get PRR [m/s]
    sigma_xy = BRF_params.sigma_xy; % standard deviation [m]
    B = BRFGaussian2D(X_B, Y_B, 1, [A, sigma_xy, [0, 0]]);   
end
d_p = size(B, 1);  % obtain the new diameter of BRF [pixel]
r_p = floor(0.5 * d_p); % radius [pixel]
BRF_params.lat_res_brf = pixel_m;   % update BRF params
BRF_params.d_pix = d_p; % update BRF params

%% 2. Define the dwell grid
[mM, nM] = size(Z_to_remove);  % get the size of full aperture

% calculate the dwell grid pixel range
dw_range.u_s = ca_range.u_s - r_p;   dw_range.u_e = ca_range.u_e + r_p;
dw_range.v_s = ca_range.v_s - r_p;   dw_range.v_e = ca_range.v_e + r_p;

% validate the dwell grid range
if(dw_range.u_s < 1 || dw_range.u_e > nM || dw_range.v_s < 1 || dw_range.v_e > mM)
    error(['Invalid clear aperture range with [' num2str(dw_range.u_s) ', ' num2str(dw_range.u_e ) ']' ' and ' '[' num2str(dw_range.v_s ) ', ' num2str(dw_range.v_e) ']']);
else
    % Dwell grid coordinates
    X_dw = X(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e );
    Y_dw = Y(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e );
    
    % Clear aperture coordinates
    X_ca = X(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Y_ca = Y(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
end

%% 3. Call RIFTA algorithm
if strcmpi(options.Algorithm, 'Iterative-FFT')
    maxIters = options.maxIters;
    RMS_dif = options.RMS_dif;
    dwellTime_dif = options.dwellTime_dif;
    
    T_dw = DwellTime2D_RIFTA_Height_IterativeFFT(...
        Z_to_remove, ...
        B, ...
        dw_range, ...
        X_ca, Y_ca, ...
        ca_range, ...
        maxIters, ...
        RMS_dif, ...
        dwellTime_dif...
        );
elseif strcmpi(options.Algorithm, 'Iterative-FFT-Optimal-DwellTime')
    T_dw = 0; ... TO DO
elseif strcmpi(options.Algorithm, 'FFT')
    T_dw = DwellTime2D_RIFTA_Height_FFT(...
        Z_to_remove, ...
        B, ...
        dw_range, ...
        ca_range...
        );
else
    error('Invalid FFT algorithm chosen. Should be either Iterative-FFT or FFT');
end
T_dw = T_dw * options.ratio;

%% 4. Downsampling the dwell grid if required
if options.isDownSampling == true
    % Obtain the sampling interval
    pixel_P_m = options.samplingInterval;
    interval_P_m = pixel_P_m / pixel_m;
    
    % Down sample the dwell grid
    X_P =  imresize(X_dw, 1/interval_P_m);
    Y_P =  imresize(Y_dw, 1/interval_P_m);
    
    % Dump X_P, Y_P & X_P_F, Y_P_F dwell point positions into a 2D array as
    %   |  u1    v1 |   P1
    %   |  u2    v2 |   P2
    %   | ...   ... |  ...
    %   |  uNt   vNt|   PNt
    P = [X_P(:), Y_P(:)];
    
    % Get the numbers of IBF machining points and sampling points of the surface error map R
    Nt = size(P, 1);
    Nr = numel(Z_to_remove);
    
    % Assemble the BRF matrix C, size(C) = Nr x Nt and vector d
    [C, d, C_T] = DwellTime2D_Height_Assemble_C_d(...
        Nr, ...
        Nt, ...
        BRF_params, ...
        Z_to_remove, ...
        X, Y, ...
        P, ...
        X_B, Y_B, ...
        B, ...
        ca_range, ...
        options.brfMode, ...
        options.viewFullAperture...
        );
    
    % Downsample T_dw
    T_P = imresize(T_dw, 1/interval_P_m, 'bicubic') * interval_P_m.^2;
    T_P = T_P - nanmin(T_P(:));
    T_P_v = T_P(:);
    
    % Clear aperture results
    Z_to_remove_ca = Z_to_remove(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Z_removal_ca = C * T_P_v;
    Z_residual_ca = d - Z_removal_ca;
    Z_removal_ca = reshape(Z_removal_ca, size(Z_to_remove_ca));
    Z_residual_ca = reshape(Z_residual_ca, size(Z_to_remove_ca));
    
    % Detilt
    Z_to_remove_ca = Z_to_remove_ca - nanmin(Z_to_remove_ca(:));
    Z_removal_ca = Z_removal_ca - nanmin(Z_removal_ca(:));
    Z_residual_ca = RemoveSurface1(X_ca, Y_ca, Z_residual_ca);
    
    if options.viewFullAperture == true
        % Full aperture results
        Z_removal = C_T * T_P_v;
        Z_residual = Z_to_remove(:) - Z_removal;
        Z_residual = Z_residual - nanmean(Z_residual);
        Z_removal = reshape(Z_removal, size(Z_to_remove));
        Z_residual = reshape(Z_residual, size(Z_to_remove));
        
        % Dwell grid results
        Z_to_remove_dw = Z_to_remove(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e );
        Z_removal_dw = Z_removal(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e );
        Z_residual_dw = Z_residual(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e );
    else
        Z_removal = 0;
        Z_residual = 0;
        
        % Dwell grid results
        Z_to_remove_dw = Z_to_remove(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e );
        Z_removal_dw = 0;
        Z_residual_dw = 0;
    end
else
    X_P = X_dw;
    Y_P = Y_dw;
    T_P = T_dw;
    
    T = zeros(size(Z_to_remove));
    T(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e ) = T_P;
    
    % Calculate the height removal in the entire aperture
    Z_removal = ConvFFT2(T,B);
    Z_residual = Z_to_remove - Z_removal;
    
    % Obtain the dwell grid result
    Z_to_remove_dw = Z_to_remove(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e );
    Z_removal_dw = Z_removal(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e );
    Z_residual_dw = Z_residual(dw_range.v_s :dw_range.v_e, dw_range.u_s:dw_range.u_e );
    
    % Obtain the clear aperture results
    Z_to_remove_ca = Z_to_remove(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Z_removal_ca = Z_removal(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Z_residual_ca = Z_residual(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    
    % De-tilt
%     Z_to_remove_ca = RemoveSurface1(X_ca, Y_ca, Z_to_remove_ca);
    Z_to_remove_ca = Z_to_remove_ca - nanmin(Z_to_remove_ca(:));
%     Z_removal_ca = RemoveSurface1(X_ca, Y_ca, Z_removal_ca);
    Z_removal_ca = Z_removal_ca - nanmin(Z_removal_ca(:));
    Z_residual_ca = RemoveSurface1(X_ca, Y_ca, Z_residual_ca);
end

end