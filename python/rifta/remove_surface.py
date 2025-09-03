import numpy as np


def remove_surface(X, Y, Z):
    # 找到 Z 中所有有限的值（不是 NaN 或 Inf）
    idx = np.isfinite(Z)
    z = Z[idx]
    x = X[idx]
    y = Y[idx]
    # 构建设计矩阵 H，包含常数项和 x、y 项
    H = np.column_stack((np.ones((x.shape[0])), x, y))
    # 使用最小二乘法求解系数 f
    f = (np.linalg.lstsq(H, z, rcond=None)[0])
    # 计算拟合平面 Zf
    # 确保 X 和 Y 的形状适合广播
    Zf = f[0] + f[1] * X + f[2] * Y
    # 计算残差 Zres
    Zres = Z - Zf
    return Zres, Zf, f