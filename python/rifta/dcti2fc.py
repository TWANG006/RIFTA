import numpy as np

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