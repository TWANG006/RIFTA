import numpy as np

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