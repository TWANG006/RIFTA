import numpy as np

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