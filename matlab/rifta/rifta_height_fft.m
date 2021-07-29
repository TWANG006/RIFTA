function Tdg = rifta_height_fft(...
    Z_to_remove, ...
    B, ... Calculated BRF
    Xdg, Ydg, dg_range,...dwell grid range
    ca_range...clear aperture range
    )
% Perform the height-based RIFTA without optimizing the total dwell time
% Output:
%   T_dw: the dwell time map on the dwell grid

%% Method 1: Calculate the dwell time by refining the inverse filtering threshold gamma
% The ca in dw range
ca_in_dw_v_s = ca_range.v_s - dg_range.v_s;
ca_in_dw_u_s = ca_range.u_s - dg_range.u_s;
ca_in_dw_v_e = ca_in_dw_v_s + ca_range.v_e - ca_range.v_s;
ca_in_dw_u_e = ca_in_dw_u_s + ca_range.u_e - ca_range.u_s;

% Calculate T for the dwell grid
Z_to_remove_dg = Z_to_remove(dg_range.v_s :dg_range.v_e, dg_range.u_s:dg_range.u_e);
Z_to_remove_dg = remove_surface(Xdg, Ydg, Z_to_remove_dg);
Z_to_remove_dg = Z_to_remove_dg - nanmin(Z_to_remove_dg(:));
Tdg = rifta_height_inverse_filter(Z_to_remove_dg, B, 1);

% Calculate the height removal in the entire aperture
Z_removal_dw = conv_fft_2d(Tdg,B); 

% Obtain the height to remove and height removal in the clear aperture
Z_to_remove_ca = Z_to_remove_dg(ca_in_dw_v_s:ca_in_dw_v_e, ca_in_dw_u_s:ca_in_dw_u_e);
Z_removal_ca = Z_removal_dw(ca_in_dw_v_s:ca_in_dw_v_e, ca_in_dw_u_s:ca_in_dw_u_e);

% Get gamma0
gamma0 = nanstd(Z_to_remove_ca(:), 1) / nanstd(Z_removal_ca(:), 1);

% Get optimized gamma
gamma = rifta_height_gamma_optimization(gamma0, Z_to_remove_dg, B, dg_range, ca_range);

% 2. Use the optimal gamma to do the inverse filtering again
Tdg = rifta_height_inverse_filter(Z_to_remove_dg, B, gamma);

end