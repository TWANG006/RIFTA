function show_rifta_height_estimation_result(...
    XB, YB, B,...TIF
    X_P, Y_P, T_P,...
    Xca, Yca, ...
    Z_to_remove_ca, Z_residual_ca, ...
    Zx_to_remove_ca, Zx_residual_ca, ...
    Zy_to_remove_ca, Zy_residual_ca ...
    )

X_B_mm = XB * 1e3;
Y_B_mm = YB * 1e3;
B_nm = B * 1e9;

% Clear aperture
X_ca_mm = Xca * 1e3;
Y_ca_mm = Yca * 1e3;
Z_to_remove_ca_nm = Z_to_remove_ca * 1e9;
Z_residual_ca_nm = Z_residual_ca * 1e9;
Zx_to_remove_ca_nm = Zx_to_remove_ca * 1e9;
Zx_residual_ca_nm = Zx_residual_ca * 1e9;
Zy_to_remove_ca_nm = Zy_to_remove_ca * 1e9;
Zy_residual_ca_nm = Zy_residual_ca * 1e9;

% Dwell time
X_P_mm = X_P * 1e3;
Y_P_mm = Y_P * 1e3;

%% TIF
subplot(3,4,1);
s = surf(X_B_mm, Y_B_mm, B_nm);
s.EdgeColor = 'none';
axis image xy; 
shading interp;
c = colorbar;
c.Label.String = 'Height [nm]';
title('TIF',...
    'FontWeight', 'normal');
xlabel('x [mm]');
ylabel('y [mm]');
view([0 90]);

%% Dwell Time 
subplot(3,4,5);
s = surf(X_P_mm, Y_P_mm, T_P);
s.EdgeColor = 'none';
axis image xy; 
shading interp;
c = colorbar;
c.Label.String = 'Time [s]';
title(['Total dwell time = ' num2str(round(sum(T_P(:)) / 60.0, 2)) ' [min], ' ],...
    'FontWeight', 'normal');
xlabel('x [mm]');
ylabel('y [mm]');
view([0 90]);

%% Clear Aperture
subplot(3,4,6);
s = surf(X_ca_mm, Y_ca_mm, Z_to_remove_ca_nm);
s.EdgeColor = 'none';
axis image xy; 
shading interp;
c = colorbar;
c.Label.String = 'Height [nm]';
title({'Height to remove in clear aperture',...
      ['PV = ' num2str(round(range(Z_to_remove_ca_nm(isfinite(Z_to_remove_ca_nm))), 2)) ' nm, ' 'RMS = ' num2str(round(nanstd(Z_to_remove_ca_nm(:), 1), 2)) ' nm']},...
    'FontWeight', 'normal');
xlabel('x [mm]');
ylabel('y [mm]');
view([0 90]);

subplot(3,4,10);
s = surf(X_ca_mm, Y_ca_mm, Z_residual_ca_nm);
s.EdgeColor = 'none';
axis image xy; 
shading interp;
pvZ = range(Z_residual_ca_nm(isfinite(Z_residual_ca_nm)));
rmsZ = nanstd(Z_residual_ca_nm(:),1);
caxis([-1 1]*3*rmsZ);
c = colorbar;
c.Label.String = 'Height [nm]';
title({'Height Residual in Clear Aperture',...
      ['PV = ' num2str(round(pvZ, 2)) ' nm, ' 'RMS = ' num2str(round(rmsZ, 2)) ' nm']},...
    'FontWeight', 'normal');
xlabel('x [mm]');
ylabel('y [mm]');
view([0 90]);

subplot(3,4,7);
s = surf(X_ca_mm, Y_ca_mm, Zx_to_remove_ca_nm);
s.EdgeColor = 'none';
axis image xy; 
shading interp;
c = colorbar;
c.Label.String = 'Slope [nrad]';
title({'X slope to remove in clear aperture',...
      ['PV = ' num2str(round(range(Zx_to_remove_ca_nm(isfinite(Zx_to_remove_ca_nm))), 2)) ' nrad, ' 'RMS = ' num2str(round(nanstd(Zx_to_remove_ca_nm(:), 1), 2)) ' nrad']},...
    'FontWeight', 'normal');
xlabel('x [mm]');
ylabel('y [mm]');
view([0 90]);

subplot(3,4,11);
s = surf(X_ca_mm, Y_ca_mm, Zx_residual_ca_nm);
s.EdgeColor = 'none';
axis image xy; 
shading interp;
pvZx = range(Zx_residual_ca_nm(isfinite(Zx_residual_ca_nm)));
rmsZx = nanstd(Zx_residual_ca_nm(:),1);
caxis([-1 1]*3*rmsZx);
c = colorbar;
c.Label.String = 'Slope [nrad]';
title({'X slope residual in Clear Aperture',...
      ['PV = ' num2str(round(pvZx, 2)) ' nrad, ' 'RMS = ' num2str(round(rmsZx, 2)) ' nrad']},...
    'FontWeight', 'normal');
xlabel('x [mm]');
ylabel('y [mm]');
view([0 90]);

subplot(3,4,8);
s = surf(X_ca_mm, Y_ca_mm, Zy_to_remove_ca_nm);
s.EdgeColor = 'none';
axis image xy; 
shading interp;
c = colorbar;
c.Label.String = 'Slope [nrad]';
title({'Y slope to remove in clear aperture',...
      ['PV = ' num2str(round(range(Zy_to_remove_ca_nm(isfinite(Zy_to_remove_ca_nm))), 2)) ' nrad, ' 'RMS = ' num2str(round(nanstd(Zy_to_remove_ca_nm(:), 1), 2)) ' nrad']},...
    'FontWeight', 'normal');
xlabel('x [mm]');
ylabel('y [mm]');
view([0 90]);

subplot(3,4,12);
s = surf(X_ca_mm, Y_ca_mm, Zy_residual_ca_nm);
s.EdgeColor = 'none';
axis image xy; 
shading interp;
pvZx = range(Zy_residual_ca_nm(isfinite(Zy_residual_ca_nm)));
rmsZx = nanstd(Zy_residual_ca_nm(:),1);
caxis([-1 1]*3*rmsZx);
c = colorbar;
c.Label.String = 'Slope [nrad]';
title({'Y slope residual in Clear Aperture',...
      ['PV = ' num2str(round(pvZx, 2)) ' nrad, ' 'RMS = ' num2str(round(rmsZx, 2)) ' nrad']},...
    'FontWeight', 'normal');
xlabel('x [mm]');
ylabel('y [mm]');
view([0 90]);


end