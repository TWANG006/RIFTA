import numpy as np

def rifta_slope_bxby_fft( X,Y,Zx_to_remove,Zy_to_remove,Bx,By,dg_range,ca_range):
    Xdg = X[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    Ydg = Y[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    Zx_to_remove_dg = Zx_to_remove[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    Zx_to_remove_dg  = Zx_to_remove_dg - np.nanmean(Zx_to_remove_dg)
    Zy_to_remove_dg = Zy_to_remove[dg_range['v_s']-1:dg_range['v_e'], dg_range['u_s']-1: dg_range['u_e']]
    Zy_to_remove_dg = Zy_to_remove_dg - np.nanmean(Zy_to_remove_dg)
    Tdg = rifta_slope_bxby_gamma_optimization(Xdg, Ydg, Zx_to_remove_dg, Zy_to_remove_dg, Bx, By, dg_range, ca_range)
    Tdg = Tdg - np.nanmin(Tdg)
    return Tdg
