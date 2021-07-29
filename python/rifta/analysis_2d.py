# Copyright (c) 2021 Tianyi Wang, Lei Huang
#
# Tianyi Wang
# tianyiwang6666@gmail.com
#
# All rights reserved
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import scipy.linalg
from pyfftw.builders import fft2
from pyfftw.builders import ifft2


def fast_fft2(A):
    fft2_obj = fft2(A, planner_effort='FFTW_ESTIMATE')
    return fft2_obj()


def fast_ifft2(A):
    ifft2_obj = ifft2(A, planner_effort='FFTW_ESTIMATE')
    return ifft2_obj()


def conv_fft_2d(S: np.ndarray, K: np.ndarray):
    '''
    Implement 2D linear convolution using FFT whose reults coincide with MATLAB `conv` "same"

    Parameters
    ----------
        S: `numpy.ndarary`
            signal
        K: `numpy.ndarray`
            kernal
    
    Returns
    -------
        `numpy.ndarary`
        The convolved signal
    '''
    mS, nS = S.shape
    mK, nK = K.shape

    # calculate padding size
    m = mS + mK - 1
    n = nS + nK - 1

    # padding S and K
    S = zero_pad_2d(S, m, n)
    K = zero_pad_2d(K, m, n)

    # crop the 'same' part
    R_crop = crop_conv_fft_2d(
        np.real(fast_ifft2(fast_fft2(S) * fast_fft2(K))), mS, nS, mK, nK
    )

    return R_crop


def crop_conv_fft_2d(R, mS, nS, mK, nK):
    hmK = mK // 2
    hnK = nK // 2
    R_crop = R[hmK: mS + hmK, hnK: nS + hnK]

    return R_crop


def zero_pad_2d(A0: np.ndarray, m: int, n: int):
    '''
    Expand `A` to the size of m x n by filling in 0

    Parameters
    ----------
        A0: `numpy.ndarray`
            input array
        m: `int`
            rows
        n: `int`
            cols
    
    Returns
    -------
        `numpy.ndarray`
        Padded array
    '''
    mA0, nA0 = A0.shape
    A = np.zeros(shape=(m, n))
    A[0: mA0, 0: nA0] = A0

    return A


def remove_surface(X: np.ndarray,
                   Y: np.ndarray,
                   Z: np.ndarray,
                   order: int = 1):
    '''
    Removes a 2D surface from the surface error map Z

    Parameters
    ----------
        X: `numpy.ndarray`
            x coordinates [m]
        Y: `numpy.ndarray`
            y coordinates [m]
        Z: `numpy.ndarray`
            surface error map [m]
        order: `int`
            order of the surface profile
    
    Returns
    -------
        Z: `numpy.ndarray`
            surface-removed surface height map [m]
        Z_fit: `numpy.ndarray`
            fitted surface height map
        fit: `function`
            fitting function
    '''

    # remove the invalid entries
    id = np.isfinite(Z)
    x = X[id].reshape(-1, 1)
    y = Y[id].reshape(-1, 1)
    z = Z[id].reshape(-1, 1)

    if 0 == order:
        H = np.column_stack(
            (np.ones_like(x))
        )
        f, _, _, _ = scipy.linalg.lstsq(H, z, check_finite=False)

        def fit(X, Y):
            return np.full(Z.shape, f[0])
        Z_fit = fit(X, Y)
        Z = Z - Z_fit
    elif 1 == order:
        H = np.column_stack(
            (np.ones_like(x), x, y)
        )
        f, _, _, _ = scipy.linalg.lstsq(H, z, check_finite=False)

        def fit(X, Y):
            return f[0] + f[1] * X + f[2] * Y
        Z_fit = fit(X, Y)
        Z = Z - Z_fit
    elif 2 == order:
        H = np.column_stack(
            (np.ones_like(x), x, y, x**2, x * y, y**2)
        )
        f, _, _, _ = scipy.linalg.lstsq(H, z, check_finite=False)

        def fit(X, Y):
            return f[0] + f[1] * X + f[2] * Y + \
                f[3] * X**2 + f[4] * X * Y + f[5] * Y**2
        Z_fit = fit(X, Y)
        Z = Z - Z_fit
    elif 3 == order:
        H = np.column_stack(
            (np.ones_like(x), x, y,
             x**2, x * y,
             y**2, x**3, x**2 * y, x * y**2, y**3)
        )
        f, _, _, _ = scipy.linalg.lstsq(H, z, check_finite=False)

        def fit(X, Y):
            return f[0] + f[1] * X + f[2] * Y + \
                f[3] * X**2 + f[4] * X * Y + f[5] * Y**2 + \
                f[6] * X**3 + f[7] * X**2 * Y + f[8] * X * Y**2 + f[9] * Y**3
        Z_fit = fit(X, Y)
        Z = Z - Z_fit
    else:
        raise ValueError('order must be in the range of [0, 3].')

    return Z, Z_fit, fit


def remove_sphere(X: np.ndarray,
                  Y: np.ndarray,
                  Z: np.ndarray):
    '''
    Removes a 2D sphere fromt the surface map Z
    
    Parameters
    ----------
        X: `numpy.ndarray`
            X coordinates [m]
        Y: `numpy.ndarray`
            Y coordinates [m]
        Z: `numpy.ndarray`
            surface error map [m]

    Returns
    -------
        Z: `numpy.ndarray`
            sphere-removed surface height map [m]
        Z_fit: `numpy.ndarray`
            fitted surface height map [m]
        f: `numpy.ndarray`
            fitted coefficients
    '''
    id = np.isfinite(Z)
    x = X[id].reshape(-1, 1)
    y = Y[id].reshape(-1, 1)
    z = Z[id].reshape(-1, 1)

    H = np.column_stack(
        (np.ones_like(x), x, y, x**2, y**2)
    )
    f, _, _, _ = scipy.linalg.lstsq(H, z, check_finite=False)
    Z_fit = f[0] + f[1] * X + f[2] * Y + f[3] * X**2 + f[4] * Y**2
    Z = Z - Z_fit

    return Z, Z_fit, f


def remove_cylinder_x(X: np.ndarray,
                      Y: np.ndarray,
                      Z: np.ndarray):
    '''
    Removes a cylinder in x direction fromt the surface map Z
    
    Parameters
    ----------
        X: `numpy.ndarray`
            x coordinates [m]
        Y: `numpy.ndarray`
            y coordinates [m]
        Z: `numpy.ndarray`
            surface error map [m]

    Returns
    -------
        Z: `numpy.ndarray`
            cylinder-removed surface height map [m]
        Z_fit: `numpy.ndarray`
            fitted surface height map
        f: `numpy.ndarray`
            fitted coefficients
    '''
    id = np.isfinite(Z)
    x = X[id].reshape(-1, 1)
    y = Y[id].reshape(-1, 1)
    z = Z[id].reshape(-1, 1)

    H = np.column_stack(
        (np.ones_like(x), x, y, x**2)
    )
    f, _, _, _ = scipy.linalg.lstsq(H, z, check_finite=False)
    Z_fit = f[0] + f[1] * X + f[2] * Y + f[3] * X**2
    Z = Z - Z_fit

    return Z, Z_fit, f
