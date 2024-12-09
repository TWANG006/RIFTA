import numpy as np
from scipy.interpolate import interp2d

def rifta_height(X,Y,Z_to_remove,tifParams, Xtif, Ytif,Ztif,ca_range,options=None):  #options参数可变
    '该函数实现了驻留时间所在的基于高度的RIFTA'
    "0.处理默认输入参数"
    defaultOptions = {
        'algorithm': 'iterative-fft',
        'tifMode': 'avg',
        'isResampling': False,
        'resamplingInterval': 1e-3,
        'ratio': 1,
        'maxIters': 10,
        'rmsDif': 0.001e-9,
        'dwellTimeDif': 60
    }
    if options is None:
        options =  defaultOptions
    "1.使用TIF参数构造TIF"
    pixel_m = np.median(np.diff(X[0, :]))                                               # 表面分辨率(X 第一行元素差值的中位数,np.diff插值np.median中位数)
    tif_r = 0.5 * tifParams['d']                                                        #TIF半径如：tifParams = {'d': 2.0}
    '在表面分辨率中生成TIF的坐标'
    #Python 中我们使用np.arange 函数用于生成从  -tif_r 作为起始点，tif_r + pixel_m 作为结束点，以确保包含 tif_r 在内的值。 np.meshgrid 函数根据这个序列生成两个坐标矩阵。
    X_B, Y_B = np.meshgrid(np.arange(-tif_r, tif_r , pixel_m),np.arange(-tif_r, tif_r , pixel_m))
    Y_B = -Y_B                                                                          #改变y方向使其向上
    if 'tifMode' in options and options['tifMode'].lower() == 'avg':
        f = interp2d(Xtif, Ytif, Ztif, kind='spline')
        B = f(X_B, Y_B)
    else:
        A = tifParams['A']                                                              #得到PRR[m/s]
        sigma_xy = tifParams['sigma_xy']                                                #标准偏差[m]
        B = tif_gaussian_height_2d(X_B, Y_B, np.array([1]), [A] + sigma_xy + [0, 0])
    d_p = B.shape[0]                                                                    #获得TIF[像素]的新直径
    r_p = np.floor(0.5 * d_p)                                                           #半径(像素), np.floor向下取整
    tifParams['lat_res_tif'] = pixel_m                                                  #更新TIF参数
    tifParams['d_pix'] = d_p                                                            #更新TIF参数
    "2.定义驻留网格"
    mM, nM = Z_to_remove.shape                                                          #获得全光圈的大小
    '计算DwellGrid （DG）像素范围'
    dg_range = {
        'u_s': int(np.floor(ca_range['u_s'] - r_p)),
        'u_e': int(np.ceil(ca_range['u_e'] + r_p)),
        'v_s': int(np.floor(ca_range['v_s'] - r_p)),
        'v_e': int(np.ceil(ca_range['v_e'] + r_p)),
    }
    '验证网格范围'
    if dg_range['u_s'] < 1 or dg_range['u_e'] > nM or dg_range['v_s'] < 1 or dg_range['v_e'] > mM:
        raise ValueError(f"Invalid clear aperture range with [{dg_range['u_s']}, {dg_range['u_e']}] and [{dg_range['v_s']}, {dg_range['v_e']}]")
    else:
        '驻留网格坐标'
        Xdg = X[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1:dg_range['u_e']]
        Ydg = Y[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1:dg_range['u_e']]
        '清晰孔径坐标'
        Xca = X[ca_range['v_s'] - 1:ca_range['v_e'], ca_range['u_s'] - 1:ca_range['u_e']]
        Yca = Y[ca_range['v_s']- 1:ca_range['v_e'], ca_range['u_s'] - 1:ca_range['u_e']]
    if 'algorithm' in options and options['algorithm'].lower() == 'iterative-fft':
        Tdg = 0
    elif 'algorithm' in options and options['algorithm'].lower() == 'iterative-fft-optimal-dwell-time':
        Tdg = 0
    elif 'algorithm' in options and options['algorithm'].lower() == 'fft':
        "3.调用RIFTA算法"
        Tdg = rifta_height_fft(Z_to_remove,B,Xdg,Ydg, dg_range,ca_range)                        #函数没导入
    else:
        raise ValueError('Invalid FFT algorithm chosen. Should be either Iterative-FFT or FFT')
    if options.get('isResampling', True):
        Tdg = 0
    else:
        Tdg = Tdg * options['ratio']
        "4. 估计"
        X_P = Xdg
        Y_P = Ydg
        T_P = Tdg
        T = np.zeros(Z_to_remove.shape)                                                 #创建一个与 Z_to_remove 大小相同的全零矩阵。
        T[dg_range['v_s'] - 1:dg_range['v_e'], dg_range['u_s'] - 1:dg_range['u_e']] = T_P
        '计算全光圈下的高度去除(二维卷积)'
        Z_removal = conv_fft_2d(T, B)
        Z_residual = Z_to_remove - Z_removal
        '获取驻留网格结果'
        Z_to_remove_dw = Z_to_remove[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e'] ]
        Z_removal_dw = Z_removal[dg_range['v_s']-1 :dg_range['v_e'], dg_range['u_s']-1:dg_range['u_e'] ]
        Z_residual_dw = Z_residual[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e'] ]
        '获取清晰光圈结果'
        Z_to_remove_ca = Z_to_remove[ca_range['v_s']-1:ca_range['v_e'], ca_range['u_s']-1:ca_range['u_e']]
        Z_removal_ca = Z_removal[ca_range['v_s']-1:ca_range['v_e'], ca_range['u_s']-1: ca_range['u_e']]
        Z_residual_ca = Z_residual[ca_range['v_s']-1:ca_range['v_e'], ca_range['u_s']-1: ca_range['u_e']]
        'De-tilt（np.nanmin(Z_to_remove_ca)求最小值）'
        Z_to_remove_ca = Z_to_remove_ca - np.nanmin(Z_to_remove_ca)
        Z_removal_ca = Z_removal_ca - np.nanmin(Z_removal_ca)
        Z_residual_ca,_,_ = remove_surface(Xca, Yca, Z_residual_ca)
    return X_B, Y_B, B, Z_removal, Z_residual,X_P, Y_P, T_P, Xdg, Ydg, dg_range, Z_to_remove_dw, Z_removal_dw, Z_residual_dw,Xca, Yca, Z_to_remove_ca, Z_removal_ca, Z_residual_ca
