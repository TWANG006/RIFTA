function [...
    X_B, Y_B, ...BRF Coordinates
    B, ... BRF
    Z_removal, Z_residual, ... full aperture results [m]
    X_P, Y_P, ...dwell grid
    T_P, ... dwell time on the dwell grid [s]
    Xdg, Ydg, ... dwll grid coordinates [m]
    dg_range, ... dwell grid range in the full aperture [m]
    Z_to_remove_dw, Z_removal_dw, Z_residual_dw,...dwell grid results [m]
    Xca, Yca, ... clear aperture coordinates [m]
    Z_to_remove_ca, Z_removal_ca, Z_residual_ca ... clear aperture results [m]
    ] = rifta_height...
    (...
    X, Y, ...full aperture coordinates [m]
    Z_to_remove,... height to remove [m]
    tifParams,... BRF parameters
    Xtif, Ytif, ...
    Ztif, ... averaged z & its coords
    ca_range, ... Clear aperture range [pixel]
    options... optgions for the algorithm
    )
% This function implements the height-based RIFTA where the dwell time is
% calculated as t * r = z

%% 0. Deal with default input arguments
defaultOptions = struct(...
    'algorithm', 'iterative-fft',...
    'tifMode', 'avg',...
    'isResampling', false,...
    'resamplingInterval', 1e-3,...
    'ratio', 1,...
    'maxIters', 10,...
    'rmsDif', 0.001e-9,...[m]
    'dwellTimeDif', 60 ...[s]
    );

if nargin == 8
    options = defaultOptions;
end

%% 1. Construct the TIF using the TIF parameters
pixel_m = median(diff(X(1, :)));
brf_r = 0.5 * tifParams.d;
[X_B, Y_B] = meshgrid(-brf_r:pixel_m:brf_r, -brf_r:pixel_m:brf_r);
Y_B = -Y_B;

if strcmpi(options.tifMode, 'avg')
    B = interp2(Xtif, Ytif, Ztif, X_B, Y_B, 'spline'); % resize BRF
else
    A = tifParams.A;   % get PRR [m/s]
    sigma_xy = tifParams.sigma_xy; % standard deviation [m]
    B = tif_gaussian_2d(X_B, Y_B, 1, [A, sigma_xy, [0, 0]]);
end
d_p = size(B, 1);  % obtain the new diameter of BRF [pixel]
r_p = floor(0.5 * d_p); % radius [pixel]
tifParams.lat_res_brf = pixel_m;   % update BRF params
tifParams.d_pix = d_p; % update BRF params

%% 2. Define the dwell grid
[mM, nM] = size(Z_to_remove);  % get the size of full aperture

% calculate the Dwell Grid (DG) pixel range
dg_range.u_s = ca_range.u_s - r_p;   dg_range.u_e = ca_range.u_e + r_p;
dg_range.v_s = ca_range.v_s - r_p;   dg_range.v_e = ca_range.v_e + r_p;

% validate the dwell grid range
if(dg_range.u_s < 1 || dg_range.u_e > nM || dg_range.v_s < 1 || dg_range.v_e > mM)
    error(['Invalid clear aperture range with [' num2str(dg_range.u_s) ', ' num2str(dg_range.u_e ) ']' ' and ' '[' num2str(dg_range.v_s ) ', ' num2str(dg_range.v_e) ']']);
else
    % Dwell grid coordinates
    Xdg = X(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    Ydg = Y(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    
    % Clear aperture coordinates
    Xca = X(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Yca = Y(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
end

%% 3. Call RIFTA algorithm
if strcmpi(options.algorithm, 'iterative-fft')
    Tdg = 0; % TO DO
elseif strcmpi(options.algorithm, 'iterative-fft-optimal-dwell-time')
    Tdg = 0; % TO DO
elseif strcmpi(options.algorithm, 'fft')
    Tdg = rifta_height_fft(...
        Z_to_remove, ...
        B, ...
        dg_range, ...
        ca_range...
        );
else
    error('Invalid FFT algorithm chosen. Should be either Iterative-FFT or FFT');
end
Tdg = Tdg * options.ratio;

%% 4. Estimation
if options.isResampling == true
    % TO DO
else
    X_P = Xdg;
    Y_P = Ydg;
    T_P = Tdg;
    
    T = zeros(size(Z_to_remove));
    T(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e ) = T_P;
    
    % Calculate the height removal in the full aperture
    Z_removal = conv_fft_2d(T, B);
    Z_residual = Z_to_remove - Z_removal;
    
    % Obtain the dwell grid result
    Z_to_remove_dw = Z_to_remove(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    Z_removal_dw = Z_removal(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    Z_residual_dw = Z_residual(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e );
    
    % Obtain the clear aperture results
    Z_to_remove_ca = Z_to_remove(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Z_removal_ca = Z_removal(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    Z_residual_ca = Z_residual(ca_range.v_s:ca_range.v_e, ca_range.u_s:ca_range.u_e);
    
    % De-tilt
    Z_to_remove_ca = Z_to_remove_ca - nanmin(Z_to_remove_ca(:));
    Z_removal_ca = Z_removal_ca - nanmin(Z_removal_ca(:));
    Z_residual_ca = remove_surface(Xca, Yca, Z_residual_ca);
end

end