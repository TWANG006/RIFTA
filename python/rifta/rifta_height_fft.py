import numpy as np

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