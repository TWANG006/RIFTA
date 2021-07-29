clear;
close all;
clc;

addpath(genpath('../rifta'));

%% Define the parameters
data_dir = '../../data/';  % data directory
surf_file = 'sim_surf_with_slopes.mat';  % surface data file
winSz = 2.5e-3;  % window size used to calculate the slopes


%% 0. Load the surface data
load([data_dir surf_file], ...
    'X', ... x coordinates
    'Y', ... y coordinates
    'Zf' ... height map without noise
    );
pixel_m = median(diff(X(1,:))); % resolution: [m/pixel], meter per pixel

%% 1. Define the Gaussian TIF (Tool Influence Function) parameters
tifParams.A = 10e-9;  % Peak Removal Rate (PRR) [m]
tifParams.lat_res_tif = pixel_m; % resolution [m/pixel]
tifParams.d = 10e-3; % diameter [m]
tifParams.d_pix = round(tifParams.d/pixel_m);  % diameter in pixels
tifParams.sigma_xy = [tifParams.d/6 tifParams.d/6];  % sigma [m]

%% 2. Define the Clear Aperture (CA)
% ca range definition [m]
min_x = nanmin(X(:));
min_y = nanmin(Y(:));
max_y = nanmax(Y(:));
ca_range_x = 190e-3;  % ca x range [m]
ca_range_y = 15e-3;   % ca y range [m]
ca_x_s = 15e-3;  % x start [m]
ca_y_s = 10e-3;  % y start [m]
ca_x_e = ca_x_s + ca_range_x;  % x end [m]
ca_y_e = ca_y_s + ca_range_y;  % y end [m]

% ca range in the unit of [pixel]
ca_range.u_s = round((ca_x_s - min_x) / pixel_m); % ca x start in pixel
ca_range.u_e = round((ca_x_e - min_x) / pixel_m); % ca x end in pixel
ca_range.v_s = round((max_y - ca_y_e) / pixel_m); % ca y start in pixel
ca_range.v_e = round((max_y - ca_y_s) / pixel_m); % ca y end in pixel

%% 3. Height-based RIFTA
% Options for height-based RIFTA
options_h = struct(...
    'algorithm', 'fft',... algorithm to use for RIFTA, 'fft' or 'iterative-fft'
    'tifMode', 'model',... mode of the TIF
    'isResampling', false, ... whether to resample to dwell grid
    'resamplingInterval', 1e-3, ...resampling interval [m/pixel]
    'ratio', 1,... ratio to be multiplied to the dwell time
    'maxIters', 20,...maximum iteration for 'iterative-fft' algorithm
    'rmsDif', 0.01e-9,...rms threshold for 'iterative-fft' algorithm
    'dwellTimeDif', 30 ...dwell time difference threshold for 'iterative-fft' algorithm [s]
    );

% call the height-based RIFTA function
[XB, YB, B, ~, ~, X_P, Y_P, T_P, ~, ~, ~, ~, ~, ~, Xca, Yca, Z_to_remove_ca, ~, Z_residual_ca] = ...
    rifta_height(...
    X,...
    Y,...
    Zf,...
    tifParams,...
    [],...
    [],...
    [],...
    ca_range,...
    options_h...
    );

% convert the height to slope
[Zx_to_remove_ca, Zy_to_remove_ca] = height_2_slopes(Xca, Yca, Z_to_remove_ca, pixel_m, winSz);
[Zx_residual_ca, Zy_residual_ca] = height_2_slopes(Xca, Yca, Z_residual_ca, pixel_m, winSz);


% display
fsfig('Height-based RIFTA without noise');
show_rifta_height_estimation_result(...
    XB, YB, B,...
    X_P, Y_P, T_P,...
    Xca, Yca, ...
    Z_to_remove_ca, Z_residual_ca, ...
    Zx_to_remove_ca, Zx_residual_ca, ...
    Zy_to_remove_ca, Zy_residual_ca ...
    );

