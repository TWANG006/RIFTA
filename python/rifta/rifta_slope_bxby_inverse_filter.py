import numpy as np

def rifta_slope_bxby_inverse_filter(  gamma,FZxy,FBxy,Coeffs):
    #print("aa\n",  gamma)
    Ny, Nx = FZxy.shape
    alpha = Coeffs / gamma
    m,n= alpha.shape
    #print("aa\n",m,n)
    #print("bb\n",alpha)
    beta = gamma / Coeffs
    beta[0, 0] = 0
    sFBxy = (FBxy * ((np.abs(FBxy * beta)) > 1)) + (alpha * ((np.abs(FBxy * beta)) <= 1))                       #remove 0's
    iFBxy = 1. / sFBxy
    iFBxy[0, 0] = 0
    Tdg = np.real(np.fft.ifft2(iFBxy* FZxy))
    Tdg = Tdg[:Ny // 2, :Nx // 2]
    return Tdg
