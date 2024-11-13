import numpy as np
from scipy.interpolate import interp2d

def rifta_slope_bxby(X, Y, Zx_to_remove, Zy_to_remove,tifParams,Xtif, Ytif, Ztifx, Ztify,ca_range, options=None):
    defaultOptions = {
        'tifMode': 'avg',
        'isResampling': False,
        'resamplingInterval': 1e-3,
        'ratio': 1
    }
    if options is None:
        options = defaultOptions
    pixel_m = np.median(np.diff(X[0, :]))  # 表面分辨率(X 第一行元素差值的中位数,np.diff插值np.median中位数)
    tif_r = 0.5 * tifParams['d']
    X_B, Y_B = np.meshgrid(np.arange(-tif_r, tif_r, pixel_m), np.arange(-tif_r, tif_r, pixel_m))
    Y_B = -Y_B
    if 'tifMode' in options and options['tifMode'].lower() == 'avg':
        f = interp2d(Xtif, Ytif, Ztifx, kind='spline')
        Bx = f(X_B, Y_B)
        f = interp2d(Xtif, Ytif, Ztify, kind='spline')
        By = f(X_B, Y_B)
    else:
        A = tifParams['A']  # 得到PRR[m/s]
        sigma_xy = tifParams['sigma_xy']
        Bx, By = tif_gaussian_slopes_2d(X_B, Y_B, np.array([1]), [A] + sigma_xy + [0, 0])
    d_p = Bx.shape[0]
    r_p = np.floor(0.5 * d_p)
    tifParams['lat_res_TIF'] = pixel_m  # 更新TIF参数
    tifParams['d_pix'] = d_p
    mM, nM = Zx_to_remove.shape
    dg_range = {
        'u_s': int(np.floor(ca_range['u_s'] - r_p)),
        'u_e': int(np.ceil(ca_range['u_e'] + r_p)),
        'v_s': int(np.floor(ca_range['v_s'] - r_p)),
        'v_e': int(np.ceil(ca_range['v_e'] + r_p)),
    }
    if dg_range['u_s'] < 1 or dg_range['u_e'] > nM or dg_range['v_s'] < 1 or dg_range['v_e'] > mM:
        raise ValueError(f"Invalid clear aperture range with [{dg_range['u_s']}, {dg_range['u_e']}] and [{dg_range['v_s']}, {dg_range['v_e']}]")
    else:
        '驻留网格坐标'
        Xdg = X[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1:dg_range['u_e']]
        Ydg = Y[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1:dg_range['u_e']]
        '清晰孔径坐标'
        Xca = X[ca_range['v_s'] - 1:ca_range['v_e'], ca_range['u_s'] - 1:ca_range['u_e']]
        Yca = Y[ca_range['v_s'] - 1:ca_range['v_e'], ca_range['u_s'] - 1:ca_range['u_e']]
    Tdg = rifta_slope_bxby_fft(X, Y, Zx_to_remove, Zy_to_remove, Bx, By, dg_range, ca_range )
    if options.get('isResampling', True):
        Tdg = 0
    else:
        Tdg = Tdg * options['ratio']
        "4. 估计"
        X_P = Xdg
        Y_P = Ydg
        T_P = Tdg
        T = np.zeros(Zx_to_remove.shape)  # 创建一个与 Z_to_remove 大小相同的全零矩阵。
        T[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1:dg_range['u_e']] = T_P
        '计算全光圈下的高度去除(二维卷积)'
        Zx_removal = conv_fft_2d(T, Bx)
        Zy_removal = conv_fft_2d(T, By)
        Zx_to_remove_ca = Zx_to_remove[ca_range['v_s'] - 1:ca_range['v_e'], ca_range['u_s'] - 1: ca_range['u_e']]
        Zx_to_remove_dg = Zx_to_remove[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1: dg_range['u_e']]
        Zy_to_remove_ca = Zy_to_remove[ca_range['v_s'] - 1:ca_range['v_e'], ca_range['u_s'] - 1: ca_range['u_e']]
        Zy_to_remove_dg = Zy_to_remove[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1: dg_range['u_e']]
        "获得整个光圈结果"
        Zx_residual = Zx_to_remove - Zx_removal
        Zy_residual = Zy_to_remove - Zy_removal
        "获取驻留网格结果"
        Zx_removal_dg = Zx_removal[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1: dg_range['u_e']]
        Zx_residual_dg = Zx_residual[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1: dg_range['u_e']]
        Zy_removal_dg = Zy_removal[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1: dg_range['u_e']]
        Zy_residual_dg = Zy_residual[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1: dg_range['u_e']]
        "获取清晰光圈结果"
        Zx_removal_ca = Zx_removal[ca_range['v_s'] - 1:ca_range['v_e'], ca_range['u_s'] - 1:ca_range['u_e']]
        Zx_residual_ca = Zx_residual[ca_range['v_s'] - 1:ca_range['v_e'], ca_range['u_s'] - 1:ca_range['u_e']]
        Zy_removal_ca = Zy_removal[ca_range['v_s'] - 1:ca_range['v_e'], ca_range['u_s'] - 1:ca_range['u_e']]
        Zy_residual_ca = Zy_residual[ca_range['v_s'] - 1:ca_range['v_e'], ca_range['u_s'] - 1:ca_range['u_e']]
        'De-tilt'
        Zx_to_remove_ca = Zx_to_remove_ca - np.nanmean(Zx_to_remove_ca)
        Zy_to_remove_ca = Zy_to_remove_ca - np.nanmean(Zy_to_remove_ca)
        Zx_removal_ca = Zx_removal_ca - np.nanmean(Zx_removal_ca)
        Zy_removal_ca = Zy_removal_ca - np.nanmean(Zy_removal_ca)
        Zx_residual_ca = Zx_residual_ca - np.nanmean(Zx_residual_ca)
        Zy_residual_ca = Zy_residual_ca - np.nanmean(Zy_residual_ca)
    return X_B, Y_B, Bx, By, Zx_removal, Zx_residual, Zy_removal, Zy_residual, X_P, Y_P, T_P, Xdg, Ydg, dg_range, Zx_to_remove_dg, Zx_removal_dg, Zx_residual_dg, Zy_to_remove_dg, Zy_removal_dg, Zy_residual_dg, Xca, Yca, Zx_to_remove_ca, Zx_removal_ca, Zx_residual_ca, Zy_to_remove_ca, Zy_removal_ca, Zy_residual_ca
