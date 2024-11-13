import numpy as np
import matplotlib.pyplot as plt

def show_rifta_slope_estimation_result(  XB, YB, Bx, By,X_P, Y_P, T_P,Xca, Yca, Z_to_remove_ca, Z_residual_ca, Zx_to_remove_ca, Zx_residual_ca, Zy_to_remove_ca, Zy_residual_ca ):
    X_B_mm = XB * 1e3
    Y_B_mm = YB * 1e3
    Bx_nrad = Bx * 1e9
    By_nrad = By * 1e9
    '通光孔径'
    X_ca_mm = Xca * 1e3
    Y_ca_mm = Yca * 1e3
    Z_to_remove_ca_nm = Z_to_remove_ca * 1e9
    Z_residual_ca_nm = Z_residual_ca * 1e9
    Zx_to_remove_ca_nm = Zx_to_remove_ca * 1e9
    Zx_residual_ca_nm = Zx_residual_ca * 1e9
    Zy_to_remove_ca_nm = Zy_to_remove_ca * 1e9
    Zy_residual_ca_nm = Zy_residual_ca * 1e9
    '保压时间'
    X_P_mm = X_P * 1e3
    Y_P_mm = Y_P * 1e3
    fig = plt.figure(figsize=(10, 10), facecolor='w')
    fig.suptitle('Slope-based RIFTA without noise')
    "TIF"
    ax = fig.add_subplot(3, 4, 1, projection='3d')  # 1
    s = ax.plot_surface(X_B_mm, Y_B_mm, Bx_nrad, cmap='viridis',shade=True)
    s.set_edgecolor('none')  # s.EdgeColor = 'none';
    # 设置轴属性,保持轴比例一致
    ax.set_box_aspect([1.5, 1.5, 1])
    # 添加颜色条
    ax.shading = 'auto'
    c = fig.colorbar(s, ax=ax)  # 将颜色条添加到子图
    c.set_label( 'Slope [nrad]')
    plt.title('x TIF', fontweight='normal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    # 设置视角
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 2, projection='3d') # 2
    s = ax.plot_surface(X_B_mm, Y_B_mm, By_nrad, cmap='viridis', shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 1.5, 1])
    ax.shading = 'auto'
    c = fig.colorbar(s, ax=ax)
    c.set_label('Slope [nrad]')
    plt.title('y TIF', fontweight='normal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    "保压时间"
    ax = fig.add_subplot(3, 4, 5, projection='3d')  # 5
    s = ax.plot_surface(X_P_mm, Y_P_mm, T_P, cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_aspect('equal')
    ax.shading = 'auto'
    c = fig.colorbar(s, ax=ax)
    c.set_label('Time [s]')
    total_dwell_time_min = str(round(np.sum(T_P) / 60.0, 2))
    plt.title(f'Total dwell time = {total_dwell_time_min} [min]', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    "通光孔径"
    ax = fig.add_subplot(3, 4, 6, projection='3d')  # 6
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Z_to_remove_ca_nm, cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.shading = 'auto'
    c = fig.colorbar(s, ax=ax)
    c.set_label('Height [nm]')
    pv = round(np.max(Z_to_remove_ca_nm[np.isfinite(Z_to_remove_ca_nm)]), 2)
    rms = round(np.nanstd(Z_to_remove_ca_nm), 2)
    # 设置标题
    plt.title('Height to remove in clear aperture\n'
              f'PV = {pv} nm, RMS = {rms} nm', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 10, projection='3d')  # 10
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Z_residual_ca_nm, cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.shading = 'auto'
    finite_Z = Z_residual_ca_nm[np.isfinite(Z_residual_ca_nm)]
    pvZ = np.max(finite_Z) - np.min(finite_Z)
    rmsZ = np.nanstd(finite_Z, ddof=1)
    #caxis_limits = [-1, 1] * 3 * rmsZ
    #plt.clim(caxis_limits[0], caxis_limits[1])
    c = fig.colorbar(s, ax=ax)
    c.set_label('Height [nm]')
    plt.title('Height Residual in Clear Aperture\n'
              f'PV = {pvZ:.2f} nm, RMS = {rmsZ:.2f} nm', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 7, projection='3d')  # 7
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Zx_to_remove_ca_nm, cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.shading = 'auto'
    c = fig.colorbar(s, ax=ax)
    c.set_label('Slope [nrad]')
    finite_Zx = Zx_to_remove_ca_nm[np.isfinite(Zx_to_remove_ca_nm)]
    pv = round(np.max(finite_Zx) - np.min(finite_Zx), 2)
    rms = round(np.nanstd(finite_Zx, ddof=1), 2)
    plt.title('X slope to remove in clear aperture\n'
              f'PV = {pv} nrad, RMS = {rms} nrad', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 11, projection='3d')  # 11
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Zx_residual_ca_nm, cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.shading = 'auto'
    finite_Zx = Zx_residual_ca_nm[np.isfinite(Zx_residual_ca_nm)]
    pvZx = np.max(finite_Zx) - np.min(finite_Zx)
    # 计算 RMS
    rmsZx = np.nanstd(finite_Zx, ddof=1)
    # 设置颜色轴的范围
    #caxis_limits = [-1, 1] * 3 * rmsZx
    #plt.clim(caxis_limits[0], caxis_limits[1])
    c = fig.colorbar(s, ax=ax)
    c.set_label('Slope [nrad]')
    plt.title('X slope residual in Clear Aperture\n'
              f'PV = {pvZx:.2f} nrad, RMS = {rmsZx:.2f} nrad', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 8, projection='3d') # 8
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Zy_to_remove_ca_nm, cmap='viridis')
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.shading = 'auto'
    c = fig.colorbar(s, ax=ax)
    c.set_label('Slope [nrad]')
    finite_Zy = Zy_to_remove_ca_nm[np.isfinite(Zy_to_remove_ca_nm)]
    pv = round(np.max(finite_Zy) - np.min(finite_Zy), 2)
    # 计算 RMS
    rms = round(np.nanstd(finite_Zy, ddof=1), 2)
    # 设置标题
    plt.title('Y slope to remove in clear aperture\n'
              f'PV = {pv} nrad, RMS = {rms} nrad', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 12, projection='3d')  # 12
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Zy_residual_ca_nm, cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.shading = 'auto'
    finite_Zx = Zy_residual_ca_nm[np.isfinite(Zy_residual_ca_nm)]
    pvZx = np.max(finite_Zx) - np.min(finite_Zx)
    # 计算 RMS
    rmsZx = np.nanstd(finite_Zx, ddof=1)
    # 设置颜色轴的范围
    #caxis_limits = [-1, 1] * 3 * rmsZx
    #plt.clim(caxis_limits[0], caxis_limits[1])
    c = fig.colorbar(s, ax=ax)
    c.set_label('Slope [nrad]')
    plt.title('Y slope residual in Clear Aperture\n'
              f'PV = {pvZx:.2f} nrad, RMS = {rmsZx:.2f} nrad', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)