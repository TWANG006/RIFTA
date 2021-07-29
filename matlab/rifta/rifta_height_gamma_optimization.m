function gamma = rifta_height_gamma_optimization(...
    gamma0, ...
    Z_to_remove_dg, ...
    B, ...
    dg_range, ...
    ca_range...
    )
% Optimize the gamma thresholding parameter used in the inverse filtering
% algorithm by minimizing f(gamma) = RMS(Z_residual_ca)
%
% Output: 
%   gamma: the optimized gamma parameter
options = optimset('TolFun', 1e-15, 'TolX', 1e-15);
fun = @(gamma)Objective_Func(gamma, Z_to_remove_dg, B, dg_range, ca_range);
gamma = fminsearch(fun, gamma0, options);

end

%% Merit function for f(gamma)
function fGamma = Objective_Func(gamma, Z_to_remove_dg, B, dw_range, ca_range)

% The ca in dw range
ca_in_dw_v_s = ca_range.v_s - dw_range.v_s + 1;
ca_in_dw_u_s = ca_range.u_s - dw_range.u_s + 1;
ca_in_dw_v_e = ca_in_dw_v_s + ca_range.v_e - ca_range.v_s;
ca_in_dw_u_e = ca_in_dw_u_s + ca_range.u_e - ca_range.u_s;

% Calculate T_dw for the dwell grid
Tdg = rifta_height_inverse_filter(...
    Z_to_remove_dg, ...
    B,...
    gamma...
    );

% Calculate the height removals in the dwell grid of x and y
Z_removal_dw = conv_fft_2d(Tdg, B);

% Calculate the height residual
Z_residual_ca = ...
    Z_to_remove_dg(ca_in_dw_v_s:ca_in_dw_v_e, ca_in_dw_u_s:ca_in_dw_u_e) - ...
    Z_removal_dw(ca_in_dw_v_s:ca_in_dw_v_e, ca_in_dw_u_s:ca_in_dw_u_e);

% Obtain the f(gamma)
fGamma = nanstd(Z_residual_ca(:), 1);

end