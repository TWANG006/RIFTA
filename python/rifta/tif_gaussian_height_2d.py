import numpy as np

def tif_gaussian_height_2d(X, Y, t, params):
    '获取参数'
    A = params[0]
    sigmax = params[1]
    sigmay = params[2]
    ux = params[3::2]
    uy = params[4::2]
    '输入结果'
    Z_fitted = np.zeros(X.shape)
    for i in range(len(t)):
        Z_fitted += A * t[i] * np.exp(-((X - ux[i]) ** 2 / (2 * sigmax ** 2) + (Y - uy[i]) ** 2 / (2 * sigmay ** 2)))
    return Z_fitted