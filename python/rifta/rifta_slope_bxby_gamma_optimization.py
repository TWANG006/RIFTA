import numpy as np
from scipy.optimize import minimize

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