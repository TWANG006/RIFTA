import numpy as np
from typing import Tuple, Dict, Any
from scipy import fft as sfft


def fft2(
    x: np.ndarray,
    s: Tuple[int, int] | None = None,
    workers: int = -1,
    norm: str = "backward",
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Universal 2D FFT forward interface (automatically chooses rfft2 or fft2).

    Parameters
    ----------
    x : ndarray
        Input real/complex array, shape (..., H, W) or (H, W).
    s : (Hpad, Wpad) or None
        Target spatial-domain size. None uses the last two dimensions of the input.
    workers : int
        Number of threads; -1 means use as many as possible.
    norm : {"backward","ortho","forward"}
        FFT normalization; default matches common convolution convention (divide by N on inverse).

    Returns
    -------
    Xhat : ndarray
        Spectrum: when using rfft2, shape (..., Hpad, Wpad//2+1); otherwise (..., Hpad, Wpad).
    info : dict
        Metadata for the inverse transform:
            - "mode": "rfft" or "c2c"
            - "s": target spatial-domain size (Hpad, Wpad)
            - "ndim": 2 (kept for consistency checks)
    """
    x = np.asarray(x)
    if s is None:
        s = tuple(map(int, x.shape[-2:]))

    is_real = not np.iscomplexobj(x)
    with sfft.set_workers(workers):
        if is_real:
            Xhat = sfft.rfft2(x, s=s, norm=norm)
            mode = "rfft"
        else:
            Xhat = sfft.fft2(x, s=s, norm=norm)
            mode = "c2c"
    info = {"mode": mode, "s": tuple(map(int, s)), "ndim": 2}
    return Xhat, info


def ifft2(
    Xhat: np.ndarray, info: Dict[str, Any], workers: int = -1, norm: str = "backward"
) -> np.ndarray:
    """
    Universal 2D FFT inverse interface (automatically chooses irfft2 or ifft2).

    Parameters
    ----------
    Xhat : ndarray
        Spectrum (from fft2u).
    info : dict
        Metadata from fft2u (contains "mode" and "s").
    workers : int
        Number of threads.
    norm : {"backward","ortho","forward"}
        Must match the forward transform (default "backward").

    Returns
    -------
    x : ndarray
        Array restored to spatial size info["s"]. If the forward used "rfft", the output is real.
    """
    mode = info.get("mode", "c2c")
    s = tuple(map(int, info.get("s")))
    if info.get("ndim", 2) != 2:
        raise ValueError("ifft2u currently only supports 2D (ndim=2).")

    with sfft.set_workers(workers):
        if mode == "rfft":
            x = sfft.irfft2(Xhat, s=s, norm=norm)
        elif mode == "c2c":
            x = sfft.ifft2(Xhat, s=s, norm=norm)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    return x


def fast_zero_padding_1d(F0, n):
    """Zero-pads a 1D array F0 to size n.
    Parameters
    ----------
    F0 : np.ndarray
        The input 1D array to be zero-padded.
    n : int
        The desired length after padding.
    Returns
    -------
    np.ndarray
        The zero-padded 1D array.
    """
    m = len(F0)
    F = np.zeros(n)
    F[:m] = F0
    return F


def fast_zero_padding_2d(F0, mm, nn):
    """Zero-pads a 2D array F0 to size (mm, nn).
    Parameters
    ----------
    F0 : np.ndarray
        The input 2D array to be zero-padded.
    mm : int
        The desired number of rows after padding.
    nn : int
        The desired number of columns after padding.
    Returns
    -------
    np.ndarray
        The zero-padded 2D array.
    """
    m, n = F0.shape
    F = np.zeros((mm, nn))
    F[:m, :n] = F0
    return F
