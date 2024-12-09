import numpy as np
from scipy.optimize import minimize

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