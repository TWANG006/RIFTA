function [...
    X_B, Y_B, Bx, By,... TIF
    Zx_removal, Zx_residual, Zy_removal, Zy_residual,... full aperture results [rad]
    X_P, Y_P, T_P,... dwell time on the dwell grid [s]
    Xdg, Ydg, dg_range, Zx_to_remove_dg, Zx_removal_dg, Zx_residual_dg, Zy_to_remove_dg, Zy_removal_dg, Zy_residual_dg,... dwell grid results [rad]
    Xca, Yca, Zx_to_remove_ca, Zx_removal_ca, Zx_residual_ca, Zy_to_remove_ca, Zy_removal_ca, Zy_residual_ca ... clear aperture results [rad]
    ] = rifta_slope_bxby(...
    X, Y, Zx_to_remove, Zy_to_remove,... slopes to remove in the full aperture [rad]
    tifParams,... BRF parameters
    Xtif, Ytif, Ztifx, Ztify,... averaged z & its coords
    ca_range,... Clear aperture range [pixel]
    options...
    )
% Purpose:
% This function implements the slope-based RIFTA algorithm where the dwell
% time is calculated as
%                       t * bx = zx
%                       t * by = zy
%
% Info:
%   Contact: tianyiwang666@gmail.com (Dr WANG Tianyi)
%   Copyright reserved.

%% 0. Set the default options for the function
defaultOptions = struct(...
    'brfMode', 'avg',...
    'isDownSampling', false,...
    'samplingInterval', 1e-3,...
    'ratio', 1...
    );

if nargin == 10
    options = defaultOptions;
end

%% 1. Construct the BRF using the BRF parameters
pixel_m = median(diff(X(1, :)));
tif_r = 0.5 * tifParams.d;
[X_B, Y_B] = meshgrid(-tif_r:pixel_m:tif_r, -tif_r:pixel_m:tif_r);
Y_B = -Y_B;

if strcmpi(options.brfMode, 'avg')
    Bx = interp2(Xtif, Ytif, Ztifx, X_B, Y_B, 'spline'); % resize BRF
    By = interp2(Xtif, Ytif, Ztify, X_B, Y_B, 'spline'); % resize BRF
else
    A = tifParams.A;   % get PRR [m/s]
    sigma_xy = tifParams.sigma_xy; % standard deviation [m]
    [Bx, By] = BRFGaussianSlope2D(X_B, Y_B, 1, [A, sigma_xy, [0, 0]]);
end
d_p = size(Bx, 1);  % obtain the new diameter of BRF [pixel]
r_p = floor(0.5 * d_p); % radius [pixel]
tifParams.lat_res_brf = pixel_m;   % update BRF params
tifParams.d_pix = d_p; % update BRF params

%% 2. Define the dwell grid & clear aperture
[mM, nM] = size(Zx_to_remove);  % get the size of full aperture

% calculate the dwell grid pixel range
dg_range.u_s = ca_range.u_s - r_p;   dg_range.u_e = ca_range.u_e + r_p;
dg_range.v_s = ca_range.v_s - r_p;   dg_range.v_e = ca_range.v_e + r_p;

% validate the dwell grid range
if(dg_range.u_s<1 || dg_range.u_e  >nM || dg_range.v_s  <1 || dg_range.v_e > mM)
    error(['Invalid clear aperture range with [' num2str(dg_range.u_s) ', ' num2str(dg_range.u_e ) ']' ' and ' '[' num2str(dg_range.v_s ) ', ' num2str(dg_range.v_e) ']']);
else
    % Dwell grid coordinates
    Xdg = X(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    Ydg = Y(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    
    % Clear aperture coordinates
    Xca = X(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Yca = Y(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
end

%% 3. Call the RIFTA slope-based dwell time algorithm
T_dw = DwellTime2D_RIFTA_Slope_bxby_FFT(...
    X,...
    Y,...
    Zx_to_remove,...
    Zy_to_remove,...
    Bx,...
    By,...
    dg_range,...
    ca_range...
    );

%% 5. Downsampling if used
% Use a sparser dwell grid
if options.isDownSampling == true
    % Obtain the sampling interval
    pixel_P_m = options.samplingInterval;
    interval_P_m = pixel_P_m / pixel_m;
    
    % Down sample the dwell grid
    X_P =  imresize(Xdg, 1/interval_P_m);
    Y_P =  imresize(Ydg, 1/interval_P_m);
    
    % Dump X_P, Y_P & X_P_F, Y_P_F dwell point positions into a 2D array as
    %   |  u1    v1 |   P1
    %   |  u2    v2 |   P2
    %   | ...   ... |  ...
    %   |  uNt   vNt|   PNt
    P = [X_P(:), Y_P(:)];
    
    % Get the numbers of IBF machining points and sampling points of the surface error map R
    Nt = size(P, 1);
    Nr = numel(Zx_to_remove);
    
    % Assemble the BRF matrix C, size(C) = Nr x Nt and vector d
    [C, d, C_T] = DwellTime2D_Assemble_C_d(Nr, Nt, tifParams, Zx_to_remove, X, Y, P, Xtif, Ytif, S_BRF, ca_range, BRF_mode, viewFullAperture);
    
    % Down sample T_dw
    T_P = imresize(T_dw, 1/interval_P_m, 'bicubic') * interval_P_m.^2;
    
    % Clear aperture results
    Zx_to_remove_ca = Zx_to_remove(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Zx_removal_ca = C * T_P_v;
    Zx_residual_ca = d - Zx_removal_ca;
    Zx_removal_ca = reshape(Zx_removal_ca, size(Zx_to_remove_ca));
    Zx_residual_ca = reshape(Zx_residual_ca, size(Zx_to_remove_ca));
    %     Z_to_remove_ca = reshape(d, size(Z_to_remove_ca));
    
    % Detilt
    Zx_to_remove_ca = RemoveSurface1(Xca, Yca, Zx_to_remove_ca);
    Zx_to_remove_ca = Zx_to_remove_ca - nanmin(Zx_to_remove_ca(:));
    Zx_removal_ca = RemoveSurface1(Xca, Yca, Zx_removal_ca);
    Zx_removal_ca = Zx_removal_ca - nanmin(Zx_removal_ca(:));
    Zx_residual_ca = RemoveSurface1(Xca, Yca, Zx_residual_ca);
    
    if viewFullAperture == true
        % Full aperture results
        Zx_removal = C_T * T_P_v;
        Zx_residual = Zx_to_remove(:) - Zx_removal;
        Zx_residual = Zx_residual - nanmean(Zx_residual);
        Zx_removal = reshape(Zx_removal, size(Zx_to_remove));
        Zx_residual = reshape(Zx_residual, size(Zx_to_remove));
        
        % Dwell grid results
        Zx_to_remove_dg = Zx_to_remove(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
        Zx_removal_dg = Zx_removal(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
        Zx_residual_dg = Zx_residual(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    else
        Zx_removal = 0;
        Zx_residual = 0;
        
        % Dwell grid results
        Zx_to_remove_dg = Zx_to_remove(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
        Zx_removal_dg = 0;
        Zx_residual_dg = 0;
    end
else
    X_P = Xdg;
    Y_P = Ydg;
    T_P = T_dw;
    
    T = zeros(size(Zx_to_remove));
    T(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e ) = T_P;
    
    % Calculate the height removal in the entire aperture
    Zx_removal = ConvFFT2(T, Bx);
    Zy_removal = ConvFFT2(T, By);
    
    % Obtain the height to remove and height removal in the clear aperture
    Zx_to_remove_ca = Zx_to_remove(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Zx_to_remove_dg = Zx_to_remove(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    Zy_to_remove_ca = Zy_to_remove(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Zy_to_remove_dg = Zy_to_remove(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    
    %% Obtain the entire aperture result
    Zx_residual = Zx_to_remove - Zx_removal;
    Zy_residual = Zy_to_remove - Zy_removal;
    
    %% Obtain the dwell grid result
    Zx_removal_dg = Zx_removal(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    Zx_residual_dg = Zx_residual(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    Zy_removal_dg = Zy_removal(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    Zy_residual_dg = Zy_residual(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    
    %% Obtain the clear aperture results
    Zx_removal_ca = Zx_removal(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Zx_residual_ca = Zx_residual(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Zy_removal_ca = Zy_removal(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Zy_residual_ca = Zy_residual(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    
    % De-tilt
    Zx_to_remove_ca = Zx_to_remove_ca - nanmean(Zx_to_remove_ca(:));
    Zy_to_remove_ca = Zy_to_remove_ca - nanmean(Zy_to_remove_ca(:));
    Zx_removal_ca = Zx_removal_ca - nanmean(Zx_removal_ca(:));
    Zy_removal_ca = Zy_removal_ca - nanmean(Zy_removal_ca(:));
    Zx_residual_ca = Zx_residual_ca - nanmean(Zx_residual_ca(:));
    Zy_residual_ca = Zy_residual_ca - nanmean(Zy_residual_ca(:));
end

end