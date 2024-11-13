import numpy as np
from scipy.io import loadmat
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
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

def rifta_height_fft( Z_to_remove,B,Xdg, Ydg, dg_range,ca_range):
    "方法1：通过细化反滤波阈值gamma计算停留时间"
    '在dw范围内的ca'
    ca_in_dw_v_s = ca_range['v_s'] - dg_range['v_s']
    ca_in_dw_u_s = ca_range['u_s'] - dg_range['u_s']
    ca_in_dw_v_e = ca_in_dw_v_s + ca_range['v_e'] - ca_range['v_s']
    ca_in_dw_u_e = ca_in_dw_u_s + ca_range['u_e'] - ca_range['u_s']
    '计算居住网格的T'
    Z_to_remove_dg = Z_to_remove[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    Z_to_remove_dg,_,_ = remove_surface(Xdg, Ydg, Z_to_remove_dg)
    Z_to_remove_dg = Z_to_remove_dg-np.nanmin(np.array(Z_to_remove_dg))
    Tdg = rifta_height_inverse_filter(Z_to_remove_dg, B, np.array([1]))
    '计算整个光圈的高度去除'
    Z_removal_dw = conv_fft_2d(Tdg, B)
    '获取要移除的高度，并在清除孔径内移除高度'
    Z_to_remove_ca = Z_to_remove_dg[ca_in_dw_v_s-1:ca_in_dw_v_e, ca_in_dw_u_s-1: ca_in_dw_u_e]
    Z_removal_ca = Z_removal_dw[ca_in_dw_v_s-1:ca_in_dw_v_e, ca_in_dw_u_s-1: ca_in_dw_u_e]
    '得到gamma0'
    gamma0 =(np.nanstd(Z_to_remove_ca))/ (np.nanstd(Z_removal_ca))                                  #np.nanstd(Z_to_remove_ca)计算 Z_to_remove_ca 的标准差，忽略 NaN 值
    '得到优化的伽马射线'
    gamma = rifta_height_gamma_optimization(gamma0, Z_to_remove_dg, B, dg_range, ca_range)
    '2.用最优的再做一次逆滤波'
    Tdg = rifta_height_inverse_filter(Z_to_remove_dg, B, gamma)
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

def rifta_height_inverse_filter(Zdg,B, gamma):
    m, n = Zdg.shape                                 #居住网格的高度大小
    B_padded = np.pad(B, ((0, m - B.shape[0]), (0, n - B.shape[1])), mode='constant')
    '执行FFT'
    FZ = np.fft.fft2(Zdg)

    FB =  np.fft.fft2(B_padded)
    '阈值'
    sFB = np.where(np.abs(FB) > 0, FB, 1 / gamma)   #np.where 函数根据 mask 的条件选择两个值中的一个：如果 mask 为 True（即 FB 中的元素非零），则选择 FB 中的元素；如果 mask 为 False（即 FB 中的元素为零），则选择 1/gamma。
    iFB = 1. / sFB

    iFB = iFB * (np.abs(sFB) * gamma > 1) + gamma * (np.abs(sFB) * gamma <= 1)
    T = np.real(np.fft.ifft2(iFB * FZ))
    '反滤波'
    T[T < 0] = 0                                    # 将 T 中所有小于 0 的元素设置为 0
    return T

def rifta_height_gamma_optimization(gamma0,Z_to_remove_dg,B,dg_range, ca_range):
    #使用 lambda 表达式来创建一个匿名函数。这个函数可以接受 gamma 作为输入参数，并调用 Objective_Func 函数，传递 gamma 和其他固定的参数。
    fun = lambda gamma: Objective_Func(gamma, Z_to_remove_dg, B, dg_range, ca_range)
    #使用 scipy.optimize.minimize 来模拟 fminsearch 的行为。method='Nelder-Mead' 方法通常用于不连续或非凸函数的优化
    result = minimize(fun, gamma0, method='Nelder-Mead')
    gamma = result.x[0]#那么 minimize 函数返回的结果对象中的 .x 属性就是这个最优的 x 值
    return gamma
def Objective_Func(gamma, Z_to_remove_dg, B, dw_range, ca_range):
    '在dw范围内的ca'
    ca_in_dw_v_s = ca_range['v_s'] - dw_range['v_s'] + 1
    ca_in_dw_u_s = ca_range['u_s'] - dw_range['u_s'] + 1
    ca_in_dw_v_e = ca_in_dw_v_s + ca_range['v_e'] - ca_range['v_s']
    ca_in_dw_u_e = ca_in_dw_u_s + ca_range['u_e'] - ca_range['u_s']
    '计算居住网格的T_dw'
    Tdg = rifta_height_inverse_filter(Z_to_remove_dg,B, gamma )
    '计算x和y的驻留网格中的高度移除'
    Z_removal_dw = conv_fft_2d(Tdg, B)
    '计算高度残差'
    Z_residual_ca = Z_to_remove_dg[ca_in_dw_v_s-1: ca_in_dw_v_e, ca_in_dw_u_s-1: ca_in_dw_u_e] - Z_removal_dw[ca_in_dw_v_s-1: ca_in_dw_v_e, ca_in_dw_u_s-1: ca_in_dw_u_e]
    '得到f（）'
    fGamma = np.nanstd(Z_residual_ca.ravel(), ddof=1)
    return fGamma

def zero_pad(F0,mm,nn):
    'size f0'
    m, n = F0.shape
    '生成一个更大的矩阵，大小为[mm nn]'
    F = np.zeros((mm, nn))
    '复制原始数据'
    F[:m, :n] = F0
    return F

def height_2_slopes(X,Y,Z, mpp,spot_sz):
    "0.获取所需的尺寸"
    m, n = Z.shape
    hf_win_sz = round(0.5 * spot_sz / mpp)                                #半个窗口大小（round小数四舍五入为整数）
    Zx = np.zeros((m, n))                                                 #生成0矩阵
    Zy = np.zeros((m, n))
    Z,_,_ = remove_surface(X, Y, Z)
    "1.通过滑动窗口计算坡度图"
    for i in np.arange(1, m + 1):  # (arange循环内的代码)
        for j in np.arange(1, n + 1):
            rows = np.arange(i - hf_win_sz, i + hf_win_sz + 1)  # 构建win的行
            cols = np.arange(j - hf_win_sz, j + hf_win_sz + 1)  # 构建win的列
            rows = rows[(rows >= 1) & (rows <= m)]  # 保证rows的范围
            cols = cols[(cols >= 1) & (cols <= n)]  # 保证cols的范围
            Xwin = X[rows[:, np.newaxis] - 1, cols - 1]  # get the window's heights
            Ywin = Y[rows[:, np.newaxis] - 1, cols - 1]  # np.newaxis 是一个用于增加数组维度的索引器。
            Zwin = Z[rows[:, np.newaxis] - 1, cols - 1]
            _, _, f = remove_surface(Xwin, Ywin, Zwin)  # 忽略前两个数值只接受第三个数值
            Zx[i - 1, j - 1] = f[1][0] # 求斜率x
            Zy[i - 1, j - 1] = f[2][0]
    return Zx, Zy

def show_rifta_height_estimation_result( XB, YB, B,X_P, Y_P, T_P,Xca, Yca,Z_to_remove_ca, Z_residual_ca,Zx_to_remove_ca, Zx_residual_ca,Zy_to_remove_ca, Zy_residual_ca ):
    #m,n=XB.shape
    #print("\n", m,n)
    #print("\n",XB)
    X_B_mm = XB * 1e3
    Y_B_mm = YB * 1e3
    B_nm = B * 1e9
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
    fig.suptitle('Height-based RIFTA without noise')
    "TIF"
    # 创建一个图形窗口和子图布局
    ax = fig.add_subplot(3, 4, 1, projection='3d')#1                                                  # 选择第一个子图
    # 绘制曲面图
    s = ax.plot_surface(X_B_mm, Y_B_mm, B_nm, cmap='viridis',shade=True)       # 你可以选择其他颜色映射
    # 移除网格线
    s.set_edgecolor('none')                                         #s.EdgeColor = 'none';
    # 设置轴属性,保持轴比例一致
    ax.set_aspect('equal')                                          #axis image xy;
    # 添加颜色条
    ax.shading = 'auto'
    c = fig.colorbar(s, ax=ax)                                      # 将颜色条添加到子图
    c.set_label('Height [nm]')
    # 设置标题和轴标签
    plt.title('TIF', fontweight='normal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    # 设置视角
    ax.view_init(elev=90, azim=0)                                   # elev: 仰角, azim: 方位角
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    "保压时间"
    # 创建一个图形窗口和3x4的子图布局，并选择第5个子图
    ax = fig.add_subplot(3, 4, 5, projection='3d')  # 5
    # 绘制曲面图
    s = ax.plot_surface(X_P_mm, Y_P_mm, T_P, cmap='viridis',shade=True)
    # 移除网格线
    s.set_edgecolor('none')
    # 设置轴属性，保持图像比例
    ax.set_aspect('equal')
    # 添加颜色条
    ax.shading = 'auto'
    cbar = fig.colorbar(s, ax=ax)
    cbar.set_label('Time [s]')
    # 计算总的停留时间并转换为分钟
    total_dwell_time_min = str(round(np.sum(T_P) / 60.0, 2))
    # 设置标题
    plt.title(f'Total dwell time = {total_dwell_time_min} [min]', fontweight='normal')
    # 设置轴标签
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    # 设置视角为从上方看
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    "通光孔径"
    ax = fig.add_subplot(3, 4, 6, projection='3d')#6
    s = ax.plot_surface(X_ca_mm, Y_ca_mm,  Z_to_remove_ca_nm, cmap='viridis',shade=True)
    s.set_edgecolor('none')
      # 使坐标轴保持纵横比和方向
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.set_aspect('auto')
    # 平滑颜色过渡
    ax.shading = 'auto'
    c = fig.colorbar(s, ax=ax)
    c.set_label('Height [nm]')
    pv = round(np.max(Z_to_remove_ca_nm[np.isfinite(Z_to_remove_ca_nm)]), 2)
    rms = round(np.nanstd(Z_to_remove_ca_nm), 2)
    # 设置标题
    plt.title('Height to remove in clear aperture\n' f'PV = {pv} nm, RMS = {rms} nm', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 10, projection='3d')#10
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Z_residual_ca_nm ,cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.set_aspect('auto')
    ax.shading = 'auto'
    finite_Z = Z_residual_ca_nm[np.isfinite(Z_residual_ca_nm)]
    pvZ = np.max(finite_Z) - np.min(finite_Z)
    rmsZ = np.nanstd(finite_Z,ddof=1)
    #caxis_limits = [-1 * 3 * rmsZ, 1 * 3 * rmsZ]
    #ax.set_zlim(caxis_limits)        #设置当前图像的颜色映射范围。
    c = fig.colorbar(s, ax=ax)
    c.set_label('Height [nm]')
    plt.title('Height Residual in Clear Aperture\n'
              f'PV = {pvZ:.2f} nm, RMS = {rmsZ:.2f} nm', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 7, projection='3d')#7
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Zx_to_remove_ca_nm, cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.set_aspect('auto')
    c = fig.colorbar(s, ax=ax)
    c.set_label('Slope [nrad]')
    finite_Zx = Zx_to_remove_ca_nm[np.isfinite(Zx_to_remove_ca_nm)]
    pv = round(np.max(finite_Zx) - np.min(finite_Zx), 2)
    # 计算 RMS
    rms = round(np.nanstd(finite_Zx, ddof=1), 2)
    # 设置标题
    plt.title('X slope to remove in clear aperture\n'
              f'PV = {pv} nrad, RMS = {rms} nrad', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 11, projection='3d')#11
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Zx_residual_ca_nm, cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.set_aspect('auto')
    finite_Zx = Zx_residual_ca_nm[np.isfinite(Zx_residual_ca_nm)]
    pvZx = np.max(finite_Zx) - np.min(finite_Zx)
    # 计算 RMS
    rmsZx = np.nanstd(finite_Zx, ddof=1)
    # 设置颜色轴的范围
    #caxis_limits = [-1 * 3 * rmsZ, 1 * 3 * rmsZ]
    #s.set_clim(caxis_limits[0], caxis_limits[1])
    c = fig.colorbar(s, ax=ax)
    c.set_label('Slope [nrad]')
    plt.title('X slope residual in Clear Aperture\n'
              f'PV = {pvZx:.2f} nrad, RMS = {rmsZx:.2f} nrad', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)
    ax = fig.add_subplot(3, 4, 8, projection='3d')#8
    s = ax.plot_surface(X_ca_mm, Y_ca_mm, Zy_to_remove_ca_nm, cmap='viridis',shade=True)
    s.set_edgecolor('none')
    ax.set_box_aspect([1.5, 0.2, 1])
    ax.set_aspect('auto')
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
    ax.set_aspect('auto')
    finite_Zx = Zy_residual_ca_nm[np.isfinite(Zy_residual_ca_nm)]
    pvZx = np.max(finite_Zx) - np.min(finite_Zx)
    # 计算 RMS
    rmsZx = np.nanstd(finite_Zx, ddof=1)
    # 设置颜色轴的范围
    #caxis_limits = [-1 * 3 * rmsZ, 1 * 3 * rmsZ]
    #ax.set_zlim(caxis_limits)
    c = fig.colorbar(s, ax=ax)
    c.set_label('Slope [nrad]')
    plt.title('Y slope residual in Clear Aperture\n'
              f'PV = {pvZx:.2f} nrad, RMS = {rmsZx:.2f} nrad', fontweight='normal')
    ax.set_xlabel('\n\n\nx [mm]')
    ax.set_ylabel('y [mm]')
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(labelcolor='w', labelbottom=False, labeltop=False)



# 定义数据目录和文件名
data_dir = '../../data/'
surf_file = 'sim_surf_with_slopes.mat'
# 构建完整的文件路径
full_path = os.path.join(data_dir, surf_file)
mat_data = loadmat(full_path)
# 定义窗口大小
winSz = 2.5e-3
X = mat_data['X']
Y = mat_data['Y']
Zf = mat_data['Zf']
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
options_h = {
    'algorithm': 'fft',
    'tifMode': 'model',
    'isResampling': False,
    'resamplingInterval': 1e-3,
    'ratio': 1,
    'maxIters': 20,
    'rmsDif': 0.01e-9,
    'dwellTimeDif': 30
}
(X_B, Y_B,B,_, _,X_P, Y_P,T_P,_, _,_,_, _, _,Xca, Yca,Z_to_remove_ca, Z_removal_ca, Z_residual_ca) = rifta_height(X, Y, Zf, tifParams, [], [], [], ca_range, options_h)
Zx_to_remove_ca, Zy_to_remove_ca = height_2_slopes(Xca, Yca,Z_to_remove_ca,pixel_m,winSz)
Zx_residual_ca, Zy_residual_ca = height_2_slopes(Xca,Yca,Z_residual_ca,pixel_m,winSz)
show_rifta_height_estimation_result(X_B, Y_B, B, X_P, Y_P, T_P, Xca, Yca,Z_to_remove_ca, Z_residual_ca,Zx_to_remove_ca, Zx_residual_ca,Zy_to_remove_ca, Zy_residual_ca)
plt.show()