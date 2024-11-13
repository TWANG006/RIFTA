import numpy as np

def remove_surface(X, Y, Z):
    # 找到 Z 中所有有限的值（不是 NaN 或 Inf）
    Z1=Z.T
    X1=X.T
    Y1=Y.T
    idx = np.isfinite(Z1)
    z = Z1[idx]
    z = z.reshape(-1, 1)
    x = X1[idx]
    x = x.reshape(-1, 1)
    y = Y1[idx]
    y = y.reshape(-1, 1)
    # 构建设计矩阵 H，包含常数项和 x、y 项
    H = np.hstack((np.ones((x.shape[0], 1)), x, y))
    # 使用最小二乘法求解系数 f
    f = np.linalg.lstsq(H, z, rcond=None)[0]
    f = f.reshape(-1, 1)
    # 计算拟合平面 Zf
    # 确保 X 和 Y 的形状适合广播
    Zf = f[0] + f[1] * X + f[2] * Y
    # 计算残差 Zres
    Zres = Z - Zf
    return Zres, Zf, f