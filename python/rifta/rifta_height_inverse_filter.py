import numpy as np

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