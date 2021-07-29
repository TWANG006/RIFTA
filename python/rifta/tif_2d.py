# Copyright (c) 2021 Tianyi Wang
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
from scipy.optimize import least_squares


def tif_surpergaussian_2d(X: np.ndarray,
                          Y: np.ndarray,
                          t: float,
                          a: float,
                          sigma: np.ndarray,
                          mu: np.ndarray = np.array([0, 0]),
                          p: float = 1.0) -> np.ndarray:
    '''
    Returns a 2D super Gaussian function

    Parameters
    ----------
        X: `numpy.ndarray`
            x coordinates [m] 
        Y: `numpy.ndarray`
            y coordinates [m]
        t: `float`
            dwell time [m]
        a: `float`
            PRR [m/s]
        sigma: `numpy.ndarray`
            variances in x and y [m]
        mu: `numpy.ndarray`
            mu[2] = center x and y
        p: `numpy.ndarray`
            power
    
    Returns
    -------
        `numpy.ndarray`
        2D Super Gaussian profile
    '''
    return t * a * np.exp(-((X - mu[0])**2 / (2 * sigma[0]**2) + (Y - mu[1])**2 / (2 * sigma[1]**2))**p)


def cost_func_tif_gaussian_2d(params: np.ndarray,
                              X: np.ndarray,
                              Y: np.ndarray,
                              Z: np.ndarray,
                              t: np.ndarray) -> np.ndarray:
    '''
    (Inner function) Return the difference between the measured and the hteoretical super Gaussian function

    Paramters
    ---------
        params: `numpy.ndarray`
            [0]: a [m/s], [1:3]: sigma_x, sigma_y [m], [3]: power=1, [4:]: centers, mu_x0, mu_y0, mu_x1, mu_y1, ...
        X: `numpy.ndarray`
            x coordinates [m], 3D array
        Y: `numpy.ndarray`
            y coordinates [m], 3D array
        Z: `numpy.ndarray`
            sets of surface errors [m], 3D array
        t: `numpy.ndarray`
            sets of dwell time [s]
    
    Returns
    -------
        `numpy.ndarray`
        Cost function for the 2D Super Gaussian profile learning
    '''
    a = params[0]
    sigma = params[1:3]
    mu = params[3:]

    Z_fit = np.zeros_like(X)
    t_v = np.atleast_1d(t)

    for i in range(len(t_v)):
        mui = mu[i * 2: i * 2 + 2]
        Z_fit[:, :, i] = tif_surpergaussian_2d(
            X[:, :, i], Y[:, :, i], t_v[i], a, sigma, mui, 1
        )

    dZ = Z.ravel() - Z_fit.ravel()
    dZ = dZ[np.isfinite(dZ)]

    return dZ

def learn_tif_gaussian_from_fps_2d(params_ini: np.ndarray,
                                   X: np.ndarray,
                                   Y: np.ndarray,
                                   Z: np.ndarray,
                                   t: np.ndarray,
                                   r_tif: float):
    ''' 
    Returns the optimized 'params_opt' for the tif, using nonlinear least-squares to fit 
    the footprints extracted from 'x' and 'z'

    Parameters
    ----------
        params_ini: `numpy.ndarray`
            [a_ini, sigma_x, sigma_y, p, mu0_ini, mu1_ini, ...]
        X: `numpy.ndarray`
            x coordinates of the 1d height profile [m]
        Y: `numpy.ndarray`
            y coordinates of the 1d height profile [m]
        Z: `numpy.ndarray`
            2D height profile [m]
        t: `numpy.ndarray`
            set of dwell times [s]
        r_tif: `float`
            radius of the tif [m]

    Returns
    -------
        params_opt: `numpy.ndarray`
            optmized params
        X_fps: `numpy.ndarray`
            2D numpy array, x coordinates of footpritns
        Y_fps: `numpy.ndarray`
            2D numpy array, y coordinates of footpritns
        Z_fps: `numpy.ndarray`
            2D numpy array, profiles of footprints
        X_tif: `numpy.ndarray`
            x tif coordinates
        Y_tif: `numpy.ndarray`
            y tif coordinates 
        Z_tif: `numpy.ndarray`
            averaged tif profiles
    '''

    n = len(np.atleast_1d(t))    # number of footpritns
    pixel_m = np.median(np.diff(X[0, :]))    # pixel size

    # half window size of tif [pixels]
    hf_win_sz = np.round(r_tif / pixel_m).astype(int)

    # preallocation of data for fitting
    X_fps = np.zeros((hf_win_sz * 2 + 1, hf_win_sz * 2 + 1, n))
    Y_fps = np.zeros((hf_win_sz * 2 + 1, hf_win_sz * 2 + 1, n))
    Z_fps = np.zeros((hf_win_sz * 2 + 1, hf_win_sz * 2 + 1, n))

    # initial guesses of the centers
    ux_ini = params_ini[3::2]
    uy_ini = params_ini[4::2]
    pix_ux_ini = np.round(
        (ux_ini - np.nanmin(X[0, :])) / pixel_m).astype(int)
    pix_uy_ini = np.round(
        (np.nanmax(Y[:, 0]) - uy_ini) / pixel_m).astype(int)

    # feed the data into the preallocated data
    for i in range(n):
        # raw guess according to the data to find the peak
        tmp = Z[np.ix_(pix_uy_ini[i] + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int),
                       pix_ux_ini[i] + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int))]
        id_max = np.unravel_index(np.nanargmax(tmp), tmp.shape)
        pix_peak_x = pix_ux_ini[i] - hf_win_sz + id_max[1] + 1
        pix_peak_y = pix_uy_ini[i] - hf_win_sz + id_max[0] + 1

        # feeding the data id_max for each footprint
        idy = pix_peak_y + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int)
        idx = pix_peak_x + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int)

        # crop & feed the data
        X_fps[:, :, i] = X[np.ix_(idy, idx)]
        Y_fps[:, :, i] = Y[np.ix_(idy, idx)]
        Z_fps[:, :, i] = Z[np.ix_(idy, idx)]
        Z_fps[:, :, i] = Z_fps[:, :, i] - np.nanmin(Z_fps[:, :, i])

    # learn the center first
    res = least_squares(
        cost_func_tif_gaussian_2d, params_ini,
        ftol=1e-15, gtol=1e-15, xtol=1e-15, max_nfev=1e6,
        args=(X_fps, Y_fps, Z_fps, t)
    )

    # learn again using the new centers mu_opt
    ux_ini = res.x[3::2]
    uy_ini = res.x[4::2]
    pix_ux_ini = np.round(
        (ux_ini - np.nanmin(X[0, :])) / pixel_m).astype(int)
    pix_uy_ini = np.round(
        (np.nanmax(Y[:, 0]) - uy_ini) / pixel_m).astype(int)

    # feed the data into the preallocated data
    for i in range(n):
        # raw guess according to the data to find the peak
        tmp = Z[np.ix_(pix_uy_ini[i] + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int),
                       pix_ux_ini[i] + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int))]
        id_max = np.unravel_index(np.nanargmax(tmp), tmp.shape)
        pix_peak_x = pix_ux_ini[i] - hf_win_sz + id_max[1] + 1
        pix_peak_y = pix_uy_ini[i] - hf_win_sz + id_max[0] + 1

        # feeding the data id_max for each footprint
        idy = pix_peak_y + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int)
        idx = pix_peak_x + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int)

        # crop & feed the data
        X_fps[:, :, i] = X[np.ix_(idy, idx)]
        Y_fps[:, :, i] = Y[np.ix_(idy, idx)]
        Z_fps[:, :, i] = Z[np.ix_(idy, idx)]
        Z_fps[:, :, i] = Z_fps[:, :, i] - np.nanmin(Z_fps[:, :, i])

    # learn the center first
    res = least_squares(
        cost_func_tif_gaussian_2d, params_ini,
        ftol=1e-15, gtol=1e-15, xtol=1e-15, max_nfev=1e6,
        args=(X_fps, Y_fps, Z_fps, t)
    )

    Z_tif = np.sum(Z_fps, axis=2) / np.sum(t)
    Z_tif[Z_tif < 0] = 0
    X_tif, Y_tif = np.meshgrid(
        np.arange(-hf_win_sz, hf_win_sz + 1),
        np.arange(-hf_win_sz, hf_win_sz + 1)
    )
    X_tif = X_tif * pixel_m
    Y_tif = -Y_tif * pixel_m

    return res.x, X_tif, Y_tif, Z_tif, X_fps, Y_fps, Z_fps

def cost_func_tif_supergaussian_2d(params: np.ndarray,
                                   X: np.ndarray,
                                   Y: np.ndarray,
                                   Z: np.ndarray,
                                   t: np.ndarray) -> np.ndarray:
    '''
    (Inner function) Return the difference between the measured and the hteoretical super Gaussian function

    Paramters
    ---------
        params: `numpy.ndarray`
            [0]: a [m/s], [1:3]: sigma_x, sigma_y [m], [3]: power, [4:]: centers, mu_x0, mu_y0, mu_x1, mu_y1, ...
        X: `numpy.ndarray`
            x coordinates [m], 3D array
        Y: `numpy.ndarray`
            y coordinates [m], 3D array
        Z: `numpy.ndarray`
            sets of surface errors [m], 3D array
        t: `numpy.ndarray`
            sets of dwell time [s]
    
    Returns
    -------
        `numpy.ndarray`
        Cost function for the 2D Super Gaussian profile learning
    '''
    a = params[0]
    sigma = params[1:3]
    p = params[3]
    mu = params[4:]

    Z_fit = np.zeros_like(X)
    t_v = np.atleast_1d(t)

    for i in range(len(t_v)):
        mui = mu[i * 2: i * 2 + 2]
        Z_fit[:, :, i] = tif_surpergaussian_2d(
            X[:, :, i], Y[:, :, i], t_v[i], a, sigma, mui, p
        )

    dZ = Z.ravel() - Z_fit.ravel()
    dZ = dZ[np.isfinite(dZ)]

    return dZ


def learn_tif_supergaussian_from_fps_2d(params_ini: np.ndarray,
                                        X: np.ndarray,
                                        Y: np.ndarray,
                                        Z: np.ndarray,
                                        t: np.ndarray,
                                        r_tif: float):
    ''' 
    Returns the optimized 'params_opt' for the tif, using nonlinear least-squares to fit 
    the footprints extracted from 'x' and 'z'

    Parameters
    ----------
        params_ini: `numpy.ndarray`
            [a_ini, sigma_x, sigma_y, p, mu0_ini, mu1_ini, ...]
        X: `numpy.ndarray`
            x coordinates of the 1d height profile [m]
        Y: `numpy.ndarray`
            y coordinates of the 1d height profile [m]
        Z: `numpy.ndarray`
            2D height profile [m]
        t: `numpy.ndarray`
            set of dwell times [s]
        r_tif: `float`
            radius of the tif [m]

    Returns
    -------
        params_opt: `numpy.ndarray`
            optmized params
        X_fps: `numpy.ndarray`
            2D numpy array, x coordinates of footpritns
        Y_fps: `numpy.ndarray`
            2D numpy array, y coordinates of footpritns
        Z_fps: `numpy.ndarray`
            2D numpy array, profiles of footprints
        X_tif: `numpy.ndarray`
            x tif coordinates
        Y_tif: `numpy.ndarray`
            y tif coordinates 
        Z_tif: `numpy.ndarray`
            averaged tif profiles
    '''

    n = len(np.atleast_1d(t))    # number of footpritns
    pixel_m = np.median(np.diff(X[0, :]))    # pixel size

    # half window size of tif [pixels]
    hf_win_sz = np.round(r_tif / pixel_m).astype(int)

    # preallocation of data for fitting
    X_fps = np.zeros((hf_win_sz * 2 + 1, hf_win_sz * 2 + 1, n))
    Y_fps = np.zeros((hf_win_sz * 2 + 1, hf_win_sz * 2 + 1, n))
    Z_fps = np.zeros((hf_win_sz * 2 + 1, hf_win_sz * 2 + 1, n))

    # initial guesses of the centers
    ux_ini = params_ini[4::2]
    uy_ini = params_ini[5::2]
    pix_ux_ini = np.round(
        (ux_ini - np.nanmin(X[0, :])) / pixel_m).astype(int)
    pix_uy_ini = np.round(
        (np.nanmax(Y[:, 0]) - uy_ini) / pixel_m).astype(int)

    # feed the data into the preallocated data
    for i in range(n):
        # raw guess according to the data to find the peak
        tmp = Z[np.ix_(pix_uy_ini[i] + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int),
                       pix_ux_ini[i] + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int))]
        id_max = np.unravel_index(np.nanargmax(tmp), tmp.shape)
        pix_peak_x = pix_ux_ini[i] - hf_win_sz + id_max[1] + 1
        pix_peak_y = pix_uy_ini[i] - hf_win_sz + id_max[0] + 1

        # feeding the data id_max for each footprint
        idy = pix_peak_y + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int)
        idx = pix_peak_x + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int)

        # crop & feed the data
        X_fps[:, :, i] = X[np.ix_(idy, idx)]
        Y_fps[:, :, i] = Y[np.ix_(idy, idx)]
        Z_fps[:, :, i] = Z[np.ix_(idy, idx)]
        Z_fps[:, :, i] = Z_fps[:, :, i] - np.nanmin(Z_fps[:, :, i])

    # learn the center first
    res = least_squares(
        cost_func_tif_supergaussian_2d, params_ini,
        ftol=1e-15, gtol=1e-15, xtol=1e-15, max_nfev=1e6,
        args=(X_fps, Y_fps, Z_fps, t)
    )

    # learn again using the new centers mu_opt
    ux_ini = res.x[4::2]
    uy_ini = res.x[5::2]
    pix_ux_ini = np.round(
        (ux_ini - np.nanmin(X[0, :])) / pixel_m).astype(int)
    pix_uy_ini = np.round(
        (np.nanmax(Y[:, 0]) - uy_ini) / pixel_m).astype(int)

    # feed the data into the preallocated data
    for i in range(n):
        # raw guess according to the data to find the peak
        tmp = Z[np.ix_(pix_uy_ini[i] + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int),
                       pix_ux_ini[i] + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int))]
        id_max = np.unravel_index(np.nanargmax(tmp), tmp.shape)
        pix_peak_x = pix_ux_ini[i] - hf_win_sz + id_max[1] + 1
        pix_peak_y = pix_uy_ini[i] - hf_win_sz + id_max[0] + 1

        # feeding the data id_max for each footprint
        idy = pix_peak_y + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int)
        idx = pix_peak_x + np.arange(-hf_win_sz, hf_win_sz + 1, 1, dtype=int)

        # crop & feed the data
        X_fps[:, :, i] = X[np.ix_(idy, idx)]
        Y_fps[:, :, i] = Y[np.ix_(idy, idx)]
        Z_fps[:, :, i] = Z[np.ix_(idy, idx)]
        Z_fps[:, :, i] = Z_fps[:, :, i] - np.nanmin(Z_fps[:, :, i])

    # learn the center first
    res = least_squares(
        cost_func_tif_supergaussian_2d, params_ini,
        ftol=1e-15, gtol=1e-15, xtol=1e-15, max_nfev=1e6,
        args=(X_fps, Y_fps, Z_fps, t)
    )

    Z_tif = np.sum(Z_fps, axis=2) / np.sum(t)
    Z_tif[Z_tif < 0] = 0
    X_tif, Y_tif = np.meshgrid(
        np.arange(-hf_win_sz, hf_win_sz + 1),
        np.arange(-hf_win_sz, hf_win_sz + 1)
    )
    X_tif = X_tif * pixel_m
    Y_tif = -Y_tif * pixel_m

    return res.x, X_tif, Y_tif, Z_tif, X_fps, Y_fps, Z_fps


def tif_superposedgaussian_2d(X: np.ndarray,
                              Y: np.ndarray,
                              t: np.ndarray,
                              params: np.ndarray) -> np.ndarray:
    '''
    Retruns a set of two superposed Gaussian functions in 2D.

    Parameters
    ----------
        X: `numpy.ndarray`
            sets of coordinates [m]
        t: `numpy.ndarray`
            sets of dwell times [s]
        params:`numpy.ndarray`
            [0, 1]: PRR [m/s], [2, 3]: sigma [m], [4,  ]: centers [m]
    '''

    # Release the parameters
    A = params[0: 2]
    sigmax = params[2: 5: 2]
    sigmay = params[3: 6: 2]
    ux = params[6::2]
    uy = params[7::2]

    Z = np.zeros_like(X)

    for i in range(len(t)):
        for j, aj in enumerate(A):
            Z[:, :, i] += t[i] * aj * np.exp(-(
                (X[:, :, i] - ux[i * 2 + j])**2 / (2 * sigmax[j]**2)
                + (Y[:, :, i] - uy[i * 2 + j])**2 / (2 * sigmay[j]**2))
            )

    return Z
