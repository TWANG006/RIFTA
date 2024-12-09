import numpy as np
from scipy.io import loadmat
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
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

def tif_gaussian_slopes_2d(X,Y,t,params):
    '获取参数'
    A = params[0]
    sigmax = params[1]
    sigmay = params[2]
    ux = params[3::2]
    uy = params[4::2]
    '输入结果'
    Zx_fitted = np.zeros(X.shape)
    Zy_fitted = np.zeros(Y.shape)
    # 计算 Zx_fitted 和 Zy_fitted
    for i in range(len(t)):
        Zx_fitted +=(ux[i]-X)/sigmax ** 2* A * t[i] * np.exp(-((X - ux[i]) ** 2 / (2 * sigmax ** 2) + (Y - uy[i]) ** 2 / (2 * sigmay ** 2)))
        Zy_fitted +=(uy[i]-Y)/sigmax ** 2* A * t[i] * np.exp(-((X - ux[i]) ** 2 / (2 * sigmax ** 2) + (Y - uy[i]) ** 2 / (2 * sigmay ** 2)))
    return Zx_fitted, Zy_fitted

def rifta_slope_bxby_fft( X,Y,Zx_to_remove,Zy_to_remove,Bx,By,dg_range,ca_range):
    Xdg = X[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    Ydg = Y[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    Zx_to_remove_dg = Zx_to_remove[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    Zx_to_remove_dg  = Zx_to_remove_dg - np.nanmean(Zx_to_remove_dg)
    Zy_to_remove_dg = Zy_to_remove[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    Zy_to_remove_dg = Zy_to_remove_dg - np.nanmean(Zy_to_remove_dg)
    Tdg = rifta_slope_bxby_gamma_optimization(Xdg, Ydg, Zx_to_remove_dg, Zy_to_remove_dg, Bx, By, dg_range, ca_range)
    Tdg = Tdg - np.nanmin(Tdg)
    return Tdg

def conv_fft_2d_crop(R, mS, nS, mK, nK):
    if mK % 2 == 1 and nK % 2 == 1:                       # 卷积核K奇数行和奇数列
        hmK = (mK - 1) // 2                               # hmK，表示卷积核在行方向上的半宽度。
        hnK = (nK - 1) // 2                               # hnK，表示卷积核在列方向上的半宽度。
        cropped_R = R[hmK:hmK + mS, hnK:hnK + nS]         #从 hmK 到 mS + hmK 行，从 hnK 到 nS + hnK 列。(不包含mS + hmK行，不包含nS + hnK列)
    elif mK % 2 == 0 and nK % 2 == 1:                     # 卷积核K偶数行和奇数列
        hmK = mK // 2
        hnK = (nK - 1) // 2
        cropped_R = R[hmK:hmK + mS, hnK:hnK + nS]
    elif mK % 2 == 1 and nK % 2 == 0:                     # 卷积核K奇数行和偶数列
        hmK = (mK - 1) // 2
        hnK = nK // 2
        cropped_R = R[hmK:hmK + mS, hnK:hnK + nS]
    else:                                                 # 卷积核K偶数行和偶数列
        hmK = mK // 2
        hnK = nK // 2
        cropped_R = R[hmK:hmK + mS, hnK:hnK + nS]
    return cropped_R

###快速傅里叶变换（FFT）算法来计算两个二维矩阵 S 和 K 的卷积
def conv_fft_2d(S, K):
    "计算FFT填充大小"
    '信号的大小'
    mS, nS = S.shape                                      #获取S,K的行数和列数。
    '内核大小'
    mK, nK = K.shape
    'Paddign大小'                                          # 代码中的mR和nR表示卷积后结果矩阵的期望行数和列数。
    mR = mS + mK - 1
    nR = nS + nK - 1
    "函数zero_pad对S,K进行0填充"
    S=zero_pad(S, mR, nR)
    K=zero_pad(K, mR, nR)
    "执行FFT和卷积"
    S_fft = np.fft.fft2(S)                            # 把填充后的S，K进行傅里叶变换将空间域转换到频率域。
    K_fft = np.fft.fft2(K)
    R_fft = S_fft * K_fft                                 # 算符将 S 和 K 的 FFT 结果逐元素相乘。
    R = np.real(np.fft.ifft2(R_fft))                      # 从频率域转换回空间域（函数来计算二维数组的逆 FFT），提取结果的实部。
    " 对R进行裁剪使R和S的尺寸一样。"
    R_crop=conv_fft_2d_crop(R,mS, nS, mK, nK)
    return R_crop

def rifta_slope_bxby_gamma_optimization(  Xdg, Ydg, Zx_to_remove_dg, Zy_to_remove_dg, Bx, By, dg_range, ca_range):
    "扩展Zx和Zy"
    # 假设 Zx_to_remove_dg 是一个 NumPy 数组(matlab中 ,水平拼接 ;垂直拼接)
    Zx_to_remove_dg_flipped = np.fliplr(-Zx_to_remove_dg)                                          # 创建负值并左右翻转
    temp_Zx_to_remove_dg = np.hstack((Zx_to_remove_dg, Zx_to_remove_dg_flipped))                   # 水平拼接
    temp_Zx_to_remove_dg_T = temp_Zx_to_remove_dg.T                                                # temp_Zx_to_remove_dg转置
    temp_Zx_to_remove_dg_flipped = np.flipud(temp_Zx_to_remove_dg)                                 # temp_Zx_to_remove_dg垂直翻转
    temp_Zx_to_remove_dg_flipped_T = temp_Zx_to_remove_dg_flipped.T                                # temp_Zx_to_remove_dg垂直翻转转置
    temp_Zx_to_remove_dg= (np.hstack((temp_Zx_to_remove_dg_T, temp_Zx_to_remove_dg_flipped_T))).T  # 水平拼接np.hstack(垂直拼接numpy.vstack)
    Zx = temp_Zx_to_remove_dg
    Zy_to_remove_dg_flipped = np.fliplr(Zy_to_remove_dg)
    temp_Zy_to_remove_dg = np.hstack((Zy_to_remove_dg, Zy_to_remove_dg_flipped))
    temp_Zy_to_remove_dg_T =temp_Zy_to_remove_dg.T
    temp_Zy_to_remove_dg_flipped = np.flipud(-(temp_Zy_to_remove_dg))
    temp_Zy_to_remove_dg_flipped_T = temp_Zy_to_remove_dg_flipped.T
    temp_Zy_to_remove_dg = (np.hstack((temp_Zy_to_remove_dg_T, temp_Zy_to_remove_dg_flipped_T))).T
    Zy = temp_Zy_to_remove_dg
    "生成fx & fy"
    '得到扩展后的斜率矩阵的大小'
    Ny, Nx = Zx.shape
    nx = np.arange(1, Nx + 1)                                                                         # matlab索引从1开始，python索引从0开始，所以+1
    ny = np.arange(1, Ny + 1)
    '得到空间域中的单位距离'
    dx =Xdg[0, 1] - Xdg[0, 0]                                                                         # matlab索引从1开始，python索引从0开始，
    dy = Ydg[1, 0] -Ydg[0, 0]
    '得到空间域中的总放大'
    L0X = dx * Nx
    L0Y = dy * Ny
    '计算谱域中的坐标。'
    u = ((nx - 1) - Nx / 2) / L0X
    v = ((ny - 1) - Ny / 2) / L0Y
    fx, fy = np.meshgrid(u, v)                                                                    # 使用 numpy.meshgrid 函数生成二维网格的坐标矩阵
    fx = np.fft.fftshift(fx)                                                                          # np.fft.fftshift 函数用于将零频率分量移动到频谱的中心
    fy = np.fft.fftshift(fy)
    "FFT of Zx, Zy and Bx, By"
    FZxy = np.fft.fft2(Zx + 1j * Zy)
    FBxy = np.fft.fft2(Bx + 1j * By, (Ny, Nx))
    Coeffs = 1j * 2 * np.pi * (fx + 1j * fy)
    gamma0 = max([abs(Coeffs[0, 1]) / abs(FBxy[0, 1]),abs(Coeffs[1, 0]) / abs(FBxy[1, 0]),abs(Coeffs[1, 1]) / abs(FBxy[1, 1])])
    "找到最优的gamma"
    fun = lambda gamma: Objective_Func(gamma,Zx_to_remove_dg, Zy_to_remove_dg, Bx, By,FZxy,FBxy,Coeffs,dg_range,ca_range )
    result = minimize(fun, gamma0, method='Nelder-Mead')
    gamma = result.x[0]
    Tdg = rifta_slope_bxby_inverse_filter( gamma, FZxy, FBxy, Coeffs)
    return Tdg

"f（gamma）的性能函数"
def Objective_Func(gamma, Zx_to_remove_dw,Zy_to_remove_dw, Bx,By,FZxy,FBxy, Coeffs,dw_range,ca_range):
    '在dw范围内的ca'
    ca_in_dw_v_s = ca_range['v_s'] - dw_range['v_s'] + 1
    ca_in_dw_u_s = ca_range['u_s'] - dw_range['u_s'] + 1
    ca_in_dw_v_e = ca_in_dw_v_s + ca_range['v_e'] - ca_range['v_s']
    ca_in_dw_u_e = ca_in_dw_u_s + ca_range['u_e']- ca_range['u_s']
    '计算驻留位置的T_dw'
    Tdg = rifta_slope_bxby_inverse_filter(gamma, FZxy, FBxy, Coeffs)
    '计算整个光圈的高度去除'
    Zx_removal_dw = conv_fft_2d(Tdg, Bx)
    Zy_removal_dw = conv_fft_2d(Tdg, By)
    '计算残差'
    Zx_residual_ca = Zx_to_remove_dw[ca_in_dw_v_s-1:ca_in_dw_v_e, ca_in_dw_u_s-1: ca_in_dw_u_e]- Zx_removal_dw[ca_in_dw_v_s-1: ca_in_dw_v_e, ca_in_dw_u_s-1: ca_in_dw_u_e]
    Zy_residual_ca = Zy_to_remove_dw[ca_in_dw_v_s-1:ca_in_dw_v_e, ca_in_dw_u_s-1: ca_in_dw_u_e]- Zy_removal_dw[ca_in_dw_v_s-1: ca_in_dw_v_e, ca_in_dw_u_s-1: ca_in_dw_u_e]
    fGamma = np.sqrt((np.nanstd(Zx_residual_ca.ravel(), axis=0)) ** 2 + (np.nanstd(Zy_residual_ca.ravel(), axis=0))** 2)
    'fGamma = nanstd([Zx_residual_ca(:);Zy_residual_ca(:)], 1)'
    'fGamma = nanstd(Zx_residual_ca(:), 1)'
    return fGamma

def zero_pad(F0,mm,nn):
    'size f0'
    m, n = F0.shape
    '生成一个更大的矩阵，大小为[mm nn]'
    F = np.zeros((mm, nn))
    '复制原始数据'
    F[:m, :n] = F0
    return F

def rifta_slope_bxby_inverse_filter(  gamma,FZxy,FBxy,Coeffs):
    #print("aa\n",  gamma)
    Ny, Nx = FZxy.shape
    alpha = Coeffs / gamma
    m,n= alpha.shape
    #print("aa\n",m,n)
    #print("bb\n",alpha)
    beta = gamma / Coeffs
    beta[0, 0] = 0
    sFBxy = (FBxy * ((np.abs(FBxy * beta)) > 1)) + (alpha * ((np.abs(FBxy * beta)) <= 1))                       #remove 0's
    iFBxy = 1. / sFBxy
    iFBxy[0, 0] = 0
    Tdg = np.real(np.fft.ifft2(iFBxy* FZxy))
    Tdg = Tdg[:Ny // 2, :Nx // 2]
    return Tdg

def dcti2fc(sx, sy, x, y):
    '再生x和sy'
    # 假设 sx 是一个 NumPy 数组(matlab中 ,水平拼接 ;垂直拼接)
    sx_flipped = np.fliplr(-sx)                 # 创建负值并左右翻转
    temp_sx = np.hstack((sx, sx_flipped))       # 水平拼接
    temp_sx_T = temp_sx.T                       # temp_sx转置
    temp_sx_flipped = np.flipud(temp_sx)        # temp_sx垂直翻转
    temp_sx_flipped_T=temp_sx_flipped.T         # temp_sx垂直翻转转置
    temp_sx = (np.hstack((temp_sx_T, temp_sx_flipped_T))).T  # 水平拼接np.hstack(垂直拼接numpy.vstack)
    sx=temp_sx
    # 假设 yx 是一个 NumPy 数组(matlab中 ,水平拼接 ;垂直拼接)
    sy_flipped = np.fliplr(sy)                  # 创建负值并左右翻转
    temp_sy = np.hstack((sy, sy_flipped))       # 水平拼接
    temp_sy_T = temp_sy.T                       # temp_sy转置
    temp_sy_flipped = np.flipud(-temp_sy)       # temp_sy垂直翻转
    temp_sy_flipped_T=temp_sy_flipped.T         # temp_sy垂直翻转转置
    temp_sy = (np.hstack((temp_sy_T, temp_sy_flipped_T))).T # 水平拼接np.hstack(垂直拼接numpy.vstack)
    sy=temp_sy
    '得到斜率矩阵的大小'
    Ny,Nx = sx.shape                            # 获得sx的尺寸（行列数）
    nx = np.arange(1, Nx + 1)                   # matlab索引从1开始，python索引从0开始，所以+1
    ny = np.arange(1, Ny + 1)
    '得到空间域中的单位距离,数据为矩阵'
    dx = x[0, 1] - x[0, 0]                      #matlab索引从1开始，python索引从0开始，
    dy = y[1, 0] - y[0, 0]
    '得到空间域中的总放大。'
    L0X = dx*Nx                                 #矩阵相乘matlab *=np.dot(dx,Nx) 矩阵各元素相乘 .*=dx*Nx
    L0Y = dy*Ny
    '计算谱域中的坐标'
    u = ((nx-1)-Nx/2)/L0X
    v = ((ny-1)-Ny/2)/L0Y
    fx, fy = np.meshgrid(u, v)              # 使用 numpy.meshgrid 函数生成二维网格的坐标矩阵
    fx = np.fft.fftshift(fx)                    #np.fft.fftshift 函数用于将零频率分量移动到频谱的中心
    fy = np.fft.fftshift(fy)
    '用FFT求DST&DCT'
    U = np.fft.fft2(sx + 1j *sy)                # 使用 numpy.fft.fft2 函数计算二维 FFT(1j 是虚数单位)
    '计算系数。'
    C = 1 / (1j * 2 * np.pi * (fx + 1j * fy))   #计算复数频率响应(2 * np.pi 是一个常数因子,fx + 1j * fy 是一个复数数组，其中 fx 实部，fy 虚部)
    '避免Inf，这会使IDCT失败'
    C[0, 0] = 0
    '计算 C{z}'
    Cz = U*C
    'Take the 2D IDCT with IFFT'
    z = np.real(np.fft.ifft2(Cz))               #函数来计算二维数组的逆 FFT，在获取实部
    z_dct = z[0:Ny//2, 0:Nx//2]                 #matlab索引从1开始，python索引从0开始,这里使用了# // 运算符它执行整数除法并返回整数结果
    #在matlab里z_dct = z(1:Ny/2,1:Nx/2)从第一行到Ny/2行从第一列到Nx/2列（包含Ny/2行，Nx/2列）
    #在python中是按索引表示从0开始z_dct = z[0:Ny//2, 0:Nx//2] 从第0索引行（第一行）到第Ny//2索引行（第Ny//2不包含）
    #matlab索引是1-2，python索引0-2(但2不包括其实是0-1索引)
    return z_dct

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



data_dir = '../../data/'
surf_file = 'sim_surf_with_slopes.mat'
full_path = os.path.join(data_dir, surf_file)
mat_data = loadmat(full_path)
winSz = 2.5e-3
X = mat_data['X']
Y = mat_data['Y']
Zfx = mat_data['Zfx']
Zfy = mat_data['Zfy']
pixel_m = np.median(np.diff(X[0, :]))#分辨率：[m/pixel]，米/像素
tifParams = {}
tifParams['A'] = 10e-9
tifParams['lat_res_tif'] = pixel_m
tifParams['d'] = 10e-3
tifParams['d_pix'] = round(tifParams['d'] / pixel_m)
tifParams['sigma_xy'] = [tifParams['d'] / 10, tifParams['d'] / 10]
min_x = np.nanmin(X)
min_y = np.nanmin(Y)
max_y = np.nanmax(Y)
ca_range_x = 190e-3
ca_range_y = 15e-3
ca_x_s = 15e-3
ca_y_s = 10e-3
ca_x_e = ca_x_s + ca_range_x
ca_y_e = ca_y_s + ca_range_y
ca_range = {}
ca_range['u_s'] = round((ca_x_s - min_x) / pixel_m)
ca_range['u_e'] = round((ca_x_e - min_x) / pixel_m)
ca_range['v_s'] = round((max_y - ca_y_e) / pixel_m)
ca_range['v_e'] = round((max_y - ca_y_s) / pixel_m)
options_s = {
    'tifMode': 'model',
    'isResampling': False,
    'resamplingInterval': 1e-3,
    'ratio': 1
}
(XB, YB, Bx, By,_, _, _, _,X_P, Y_P, T_P,_, _, _, _, _, _, _, _, _,Xca, Yca,Zx_to_remove_ca, _, Zx_residual_ca,Zy_to_remove_ca, _, Zy_residual_ca) = rifta_slope_bxby(X, Y, Zfx, Zfy,tifParams, [],[], [], [],ca_range,options_s)
Z_to_remove_ca = dcti2fc(Zx_to_remove_ca, Zy_to_remove_ca, Xca, Yca)
Z_to_remove_ca =Z_to_remove_ca-  np.nanmin(Z_to_remove_ca)
Z_residual_ca = dcti2fc(Zx_residual_ca, Zy_residual_ca, Xca, Yca)
show_rifta_slope_estimation_result(XB, YB, Bx, By, X_P, Y_P, T_P,Xca, Yca, Z_to_remove_ca, Z_residual_ca, Zx_to_remove_ca, Zx_residual_ca,Zy_to_remove_ca, Zy_residual_ca )

plt.show()