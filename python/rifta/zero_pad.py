import numpy as np

def zero_pad(F0,mm,nn):
    'size f0'
    m, n = F0.shape
    '生成一个更大的矩阵，大小为[mm nn]'
    F = np.zeros((mm, nn))
    '复制原始数据'
    F[:m, :n] = F0
    return F