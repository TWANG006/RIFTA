import numpy as np

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