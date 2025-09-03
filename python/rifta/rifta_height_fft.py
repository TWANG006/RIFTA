import numpy as np
from conv import conv_fft_2d, nanconv_fft, FFTConvolver2D


def rifta_height_fft( Z_to_remove,B,Xdg, Ydg, dg_range,ca_range):
    "方法1：通过细化反滤波阈值gamma计算停留时间"
    '在dw范围内的ca'
    ca_in_dw_v_s = ca_range['v_s'] - dg_range['v_s']+1
    ca_in_dw_u_s = ca_range['u_s'] - dg_range['u_s']+1
    ca_in_dw_v_e = ca_in_dw_v_s + ca_range['v_e'] - ca_range['v_s']
    ca_in_dw_u_e = ca_in_dw_u_s + ca_range['u_e'] - ca_range['u_s']
    '计算居住网格的T'
    Z_to_remove_dg = Z_to_remove[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    # 找到 Z 中所有有限的值（不是 NaN 或 Inf）
    idx = np.isfinite(Z_to_remove_dg)
    z = Z_to_remove_dg[idx]
    x = Xdg[idx]
    y = Ydg[idx]
    # 构建设计矩阵 H，包含常数项和 x、y 项
    H = np.column_stack((np.ones((x.shape[0])), x, y))
    # 使用最小二乘法求解系数 f
    f = (np.linalg.lstsq(H, z, rcond=None)[0])
    # 计算拟合平面 Zf
    # 确保 X 和 Y 的形状适合广播
    Zf = f[0] + f[1] * Xdg + f[2] * Ydg
    # 计算残差 Zres
    Z_to_remove_dg = Z_to_remove_dg - Zf
    Z_to_remove_dg = Z_to_remove_dg-np.nanmin(np.array(Z_to_remove_dg))
    W = np.zeros_like(Z_to_remove_dg)
    W[ca_in_dw_v_s-1:ca_in_dw_v_e, ca_in_dw_u_s-1: ca_in_dw_u_e] = 1
    n = 3
    Aex = []
    current_Z_power = np.ones_like(Z_to_remove_dg)
    convolver = FFTConvolver2D(B, data_shape=Z_to_remove_dg.shape, workers=8)
    for _ in range(n):
        A = convolver.convolve(current_Z_power)
        A = A[W == 1]
        Aex.append(A)
        current_Z_power = current_Z_power * Z_to_remove_dg
    Aex = Aex[::-1]
    # 提取 Z_to_remove_dg 中 W 矩阵值为 1 的位置的值
    b = Z_to_remove_dg[W == 1]
    # 将 Aex 转换为二维数组并求解线性最小二乘问题
    Aex_array = (np.array(Aex)).T
    gamma_all, _, _, _ = np.linalg.lstsq(Aex_array, b, rcond=None)
    # 将 gamma_all 转换为一维数组
    p = gamma_all.flatten()
    # 使用多项式求值
    Tdg = np.polyval(p, Z_to_remove_dg)
    return Tdg