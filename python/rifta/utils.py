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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_clear_aperture_rect(X: np.ndarray,
                            Y: np.ndarray,
                            ca_range_x: float,
                            ca_range_y: float,
                            ca_center_x: float,
                            ca_center_y: float):
    '''
    Returns the clear aperture range in [m] and [pixel]
    '''
    min_x = np.nanmin(X)
    max_y = np.nanmin(Y)
    pixel_m = np.median(np.diff(X[0, :]))

    ca_xs = ca_center_x - 0.5 * ca_range_x
    ca_xe = ca_xs + ca_range_x
    ca_ys = ca_center_y - 0.5 * ca_range_y
    ca_ye = ca_ys + ca_range_y

    ca_range = {
        'u_s': round((ca_xs - min_x) / pixel_m),
        'u_e': round((ca_xe - min_x) / pixel_m) + 1,
        'v_s': round((max_y - ca_ye) / pixel_m),
        'v_e': round((max_y - ca_ys) / pixel_m) + 1
    }

    return ca_range


def get_polynomial_orders_rect(m, n):
    '''
    Returns the polynomial order grid based on the orders m, n in x, y direction, respectively
    '''
    return np.meshgrid(
        np.arange(0, n + 1),
        np.arange(0, m + 1)
    )


def normalize_xy_unit_square(X: np.ndarray, Y: np.ndarray):
    '''
    Returns the normalized X, Y coordinates within a unit sqaure, i.e [-1, 1]
    '''
    return \
        -1 + 2 * (X - np.min(X)) / (np.max(X) - np.min(X)), \
        -1 + 2 * (Y - np.min(Y)) / (np.max(Y) - np.min(Y))


def rms(a, axis=None):
    '''
    Return the RMS value of `a` along `axis` excluding NaNs

    Parameters
    ----------
        a: `array-like`
        axis: `int`
            rms along the axis, if ``None``, rms for all the elements in a
    '''
    return np.nanstd(a=a, axis=axis, ddof=1)


def pv(a, axis=None):
    '''
    Returns the PV value of `a` along `axis` exluding NaNs

    Parameters
    ----------
        a: `numpy.ndarray`
            array
        axis: `int`
            PV along the axis, if ``None``, PV for all the elements in a
    Returns
    -------
        `float`
            PV of `a` along `axis`
    '''
    return np.nanmax(a, axis) - np.nanmin(a, axis)


def show_surface_map(X: np.ndarray,
                     Y: np.ndarray,
                     Z: np.ndarray,
                     ax=None,
                     vmin=None,
                     vmax=None,
                     fig_title: str = ''):
    '''
    Display the surface error map

    Parameters
    ----------
        X: `numpy.ndarray`
            x coordinates [m]
        Y: `numpy.ndarray`
            y coordinates [m]
        Z: `numpy.ndarray`
            surface errors [m]
        fig_title: `str`
            title of the figure
    '''

    X_mm = X * 1e3
    Y_mm = Y * 1e3
    Z_nm = Z * 1e9

    rms_Z_nm = rms(Z_nm)
    pv_Z_nm = pv(Z_nm)

    if ax == None:
        fig, ax = plt.subplots()

    c = ax.pcolormesh(X_mm, Y_mm, Z_nm, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(
        fig_title +
        ': PV = {:.2f} nm, RMS = {:.2f} nm'.format(pv_Z_nm, rms_Z_nm)
    )
    ax.set_aspect('equal')
    ax.set_rasterized(True)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(c, ax=ax, cax=cax)
    cbar.set_label('height [nm]')


def show_dwell_time_map(X: np.ndarray,
                        Y: np.ndarray,
                        T: np.ndarray,
                        ax=None,
                        fig_title: str = ''):
    '''
    Display the surface error map

    Parameters
    ----------
        X: `numpy.ndarray`
            x coordinates [m]
        Y: `numpy.ndarray`
            y coordinates [m]
        T: `numpy.ndarray`
            dwell time [s]
        fig_title: `str`
            title of the figure
    '''

    X_mm = X * 1e3
    Y_mm = Y * 1e3

    total_dt = np.nansum(T) / 60.0

    if ax == None:
        fig, ax = plt.subplots()

    ax.autoscale(enable=True, axis='both', tight=True)
    c = ax.scatter(X_mm.ravel(), Y_mm.ravel(),
                   c=T.ravel(), cmap='jet')
    # c = ax.pcolormesh(X_mm, Y_mm, T, cmap='gist_jet', shading='auto')
    ax.set_title(fig_title + ' Total dwell time = {:.2f} min'.format(total_dt))
    ax.set_aspect('equal')
    ax.set_rasterized(True)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(c, ax=ax, cax=cax)
    cbar.set_label('time [s]')


def k_smallest_index(a, k):
    '''
    Returns the ``k`` smallest index in a

    Parameters
    ----------
        a: `array-like`
        k: `int`

    Returns
    -------
        `array-like`
        The ``k`` smallest index in a
    '''
    idx = np.argpartition(a.ravel(), k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))


def k_largest_index(a, k):
    '''
    Returns the ``k`` largest index in a

    Parameters
    ----------
        a: `array-like`
        k: `int`

    Returns
    -------
        `array-like`
        The ``k`` largest index in a
    '''
    idx = np.argpartition(a.ravel(), a.size-k)[-k:]
    return np.column_stack(np.unravel_index(idx, a.shape))


def fwhm_2_sigma(fwhm):
    '''
    Returns sigma from fwhm

    Parameters
    ----------
        fwhm: `array-like`

    Returns
    -------
        `array-like`
        The ``sigma`` converted from ``fwhm``
    '''
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def sigma_2_fwhm(sigma):
    '''
    Returns fwhm from sigma

    Parameters
    ----------
        sigma: `array-like`

    Returns
    -------
        `array-like`
        The ``fwhm`` converted from ``sigma``
    '''
    return 2 * np.sqrt(2 * np.log(2)) * sigma
