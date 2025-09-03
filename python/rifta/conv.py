import numpy as np
from typing import Tuple
from scipy import fft as sfft
from fft_tools import fft2, ifft2

# ----------------------------- helpers -----------------------------


def _fast_full_shape_2d(a_shape, k_shape):
    """
    Compute FFT-friendly padded shape and the true full convolution shape.

    Returns
    -------
    (Hp, Wp): tuple[int, int]
        Padded shape using scipy.fft.next_fast_len for speed.
    (Hf, Wf): tuple[int, int]
        True full convolution shape (Ha+Hk-1, Wa+Wk-1).
    """
    Ha, Wa = int(a_shape[-2]), int(a_shape[-1])
    Hk, Wk = int(k_shape[-2]), int(k_shape[-1])
    Hf, Wf = Ha + Hk - 1, Wa + Wk - 1
    Hp, Wp = sfft.next_fast_len(Hf), sfft.next_fast_len(Wf)
    return (Hp, Wp), (Hf, Wf)


def _crop_same(R_full, mS, nS, mK, nK):
    """
    Crop the center region from a full convolution to get SAME-sized output.
    Handles even/odd kernel sizes along each dimension.
    """
    if mK % 2 == 1 and nK % 2 == 1:
        hmK = (mK - 1) // 2
        hnK = (nK - 1) // 2
    elif mK % 2 == 0 and nK % 2 == 1:
        hmK = mK // 2
        hnK = (nK - 1) // 2
    elif mK % 2 == 1 and nK % 2 == 0:
        hmK = (mK - 1) // 2
        hnK = nK // 2
    else:
        hmK = mK // 2
        hnK = nK // 2
    return R_full[hmK : hmK + mS, hnK : hnK + nS]


def conv_fft_2d(S: np.ndarray, K: np.ndarray, workers: int = -1) -> np.ndarray:
    """
    2D linear convolution (SAME size) using fft2/ifft2 that auto-selects real/complex paths.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    (Hp, Wp), (Hf, Wf) = _fast_full_shape_2d(S.shape, K.shape)
    mS, nS = S.shape
    mK, nK = K.shape

    Xhat, infoX = fft2(S, s=(Hp, Wp), workers=workers)
    Khat, infoK = fft2(K, s=(Hp, Wp), workers=workers)

    # If paths differ (one rfft, one c2c), switch both to c2c to avoid shape mismatch
    if infoX["mode"] != infoK["mode"]:
        # Force recomputation as c2c spectra
        with sfft.set_workers(workers):
            if infoX["mode"] == "rfft":
                Xhat = sfft.fft2(S, s=(Hp, Wp))
            if infoK["mode"] == "rfft":
                Khat = sfft.fft2(K, s=(Hp, Wp))
        info = {"mode": "c2c", "s": (Hp, Wp), "ndim": 2}
    else:
        info = infoX  # Reuse when both modes match

    Yhat = Xhat * Khat
    full = ifft2(Yhat, info, workers=workers)
    full = full[:Hf, :Wf]
    # SAME crop
    out = _crop_same(full, mS, nS, mK, nK)

    # If inputs are real, the theoretical result is real; drop tiny imaginary noise
    if not np.iscomplexobj(S) and not np.iscomplexobj(K):
        out = out.real
    return out


def nanconv_fft(
    s: np.ndarray, k: np.ndarray, workers: int = -1, norm: str = "backward"
) -> np.ndarray:
    """
    NaN-aware convolution (SAME size) using the common fft2/ifft2 interface; auto-align rfft/c2c.
    """
    s = np.asarray(s)
    k = np.asarray(k)
    mS, nS = s.shape
    mK, nK = k.shape

    # Padded FFT shape (fast length) and true full-convolution size
    (Hp, Wp), (Hf, Wf) = _fast_full_shape_2d(s.shape, k.shape)
    s_hw = (Hp, Wp)

    # Choose frequency-domain mode: complex kernel -> c2c; otherwise prefer rfft
    mode = "c2c" if np.iscomplexobj(k) else "rfft"
    inv_info = {"mode": mode, "s": s_hw, "ndim": 2}

    # Handle NaNs in input by zeroing them and tracking valid samples
    nan_mask = np.isnan(s)
    ones = np.ones_like(s, dtype=np.float64)
    s_zero = s.copy()
    s_zero[nan_mask] = 0.0
    on = ones.copy()
    on[nan_mask] = 0.0

    # Forward transforms aligned to the chosen mode for kernel / ones / valid-mask / zeroed data
    Khat, _ = fft2(k, s=s_hw, workers=workers, norm=norm)
    Ohat, _ = fft2(ones, s=s_hw, workers=workers, norm=norm)
    X_on, _ = fft2(on, s=s_hw, workers=workers, norm=norm)
    X_sz, _ = fft2(s_zero, s=s_hw, workers=workers, norm=norm)

    # conv_on: count of valid samples; conv_o: normalization by full support; conv_s: data sum
    conv_on_full = ifft2(X_on * Khat, inv_info, workers=workers, norm=norm)
    conv_o_full = ifft2(Ohat * Khat, inv_info, workers=workers, norm=norm)
    conv_s_full = ifft2(X_sz * Khat, inv_info, workers=workers, norm=norm)

    # Trim to full size and crop to SAME
    conv_on = _crop_same(conv_on_full[:Hf, :Wf], mS, nS, mK, nK)
    conv_o = _crop_same(conv_o_full[:Hf, :Wf], mS, nS, mK, nK)
    conv_s = _crop_same(conv_s_full[:Hf, :Wf], mS, nS, mK, nK)

    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.divide(
            conv_on, conv_o, out=np.zeros_like(conv_on), where=conv_o != 0
        )
        c = np.divide(conv_s, weight, out=np.zeros_like(conv_s), where=weight != 0)

    c[nan_mask] = np.nan
    # If both s and k are real, drop tiny imaginary parts from numerical error
    if not np.iscomplexobj(s) and not np.iscomplexobj(k):
        c = c.real
    return c


# ------------------------- class interface -------------------------


class FFTConvolver2D:
    """
    Convolver using the common FFT interface with precomputed kernel spectrum.
    - Automatically selects rfft2/fft2 based on input type.
    - Precomputes Khat and Ohat (spectrum of all-ones) for reuse in convolve/nan_convolve.
    """

    def __init__(
        self, kernel: np.ndarray, data_shape: Tuple[int, int], workers: int = -1
    ):
        kernel = np.asarray(kernel)
        self.data_shape = tuple(map(int, data_shape))
        self.kernel_shape = kernel.shape
        self.workers = workers

        (Hp, Wp), (Hf, Wf) = _fast_full_shape_2d(self.data_shape, self.kernel_shape)
        self.Hp, self.Wp = Hp, Wp
        self.Hf, self.Wf = Hf, Wf

        self.Khat, self.Kinfo = fft2(kernel, s=(Hp, Wp), workers=self.workers)
        ones = np.ones(self.data_shape, dtype=np.float64)
        self.Ohat, self.Oinfo = fft2(ones, s=(Hp, Wp), workers=self.workers)

        # If paths differ, unify to c2c (simplifies elementwise products and reuse)
        if self.Kinfo["mode"] != self.Oinfo["mode"]:
            with sfft.set_workers(self.workers):
                self.Khat = sfft.fft2(kernel, s=(Hp, Wp))
                self.Ohat = sfft.fft2(ones, s=(Hp, Wp))
            self.info = {"mode": "c2c", "s": (Hp, Wp), "ndim": 2}
        else:
            self.info = dict(self.Kinfo)

    def convolve(self, S: np.ndarray) -> np.ndarray:
        # Convolve S with the precomputed kernel; returns SAME-sized result
        S = np.asarray(S)
        mS, nS = self.data_shape
        mK, nK = self.kernel_shape

        Xhat, Xinfo = fft2(S, s=(self.Hp, self.Wp), workers=self.workers)
        # If path differs from precomputed info, switch to c2c
        if Xinfo["mode"] != self.info["mode"]:
            with sfft.set_workers(self.workers):
                Xhat = sfft.fft2(S, s=(self.Hp, self.Wp))
            info = {"mode": "c2c", "s": (self.Hp, self.Wp), "ndim": 2}
        else:
            info = self.info

        Yhat = Xhat * self.Khat
        full = ifft2(Yhat, info, workers=self.workers)[: self.Hf, : self.Wf]
        out = _crop_same(full, mS, nS, mK, nK)
        # Real input + real kernel -> real output (drop tiny imaginary noise)
        if not np.iscomplexobj(S) and not np.iscomplexobj(self.Khat):
            out = out.real
        return out

    def nan_convolve(self, S: np.ndarray) -> np.ndarray:
        # Convolve with NaN handling; returns SAME-sized result
        S = np.asarray(S)
        mS, nS = self.data_shape
        mK, nK = self.kernel_shape
        nan_mask = np.isnan(S)

        s_zero = S.copy()
        s_zero[nan_mask] = 0.0

        on = np.ones(self.data_shape, dtype=np.float64)
        on[nan_mask] = 0.0

        # Path alignment: if stored info is c2c, make all spectra c2c
        def _fwd_align(A):
            Ah, Ai = fft2(A, s=(self.Hp, self.Wp), workers=self.workers)
            if Ai["mode"] != self.info["mode"]:
                with sfft.set_workers(self.workers):
                    Ah = sfft.fft2(A, s=(self.Hp, self.Wp))
                Ai = {"mode": "c2c", "s": (self.Hp, self.Wp), "ndim": 2}
            return Ah, Ai

        X_on, info_on = _fwd_align(on)
        X_zero, info_zero = _fwd_align(s_zero)

        conv_on_full = ifft2(X_on * self.Khat, self.info, workers=self.workers)[
            : self.Hf, : self.Wf
        ]
        conv_o_full = ifft2(self.Ohat * self.Khat, self.info, workers=self.workers)[
            : self.Hf, : self.Wf
        ]
        conv_s_full = ifft2(X_zero * self.Khat, self.info, workers=self.workers)[
            : self.Hf, : self.Wf
        ]

        conv_on = _crop_same(conv_on_full, mS, nS, mK, nK)
        conv_o = _crop_same(conv_o_full, mS, nS, mK, nK)
        conv_s = _crop_same(conv_s_full, mS, nS, mK, nK)

        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.divide(
                conv_on, conv_o, out=np.zeros_like(conv_on), where=conv_o != 0
            )
            result = np.divide(
                conv_s, weight, out=np.zeros_like(conv_s), where=weight != 0
            )

        result[nan_mask] = np.nan
        # Real input (excluding NaNs) + real kernel -> real output
        if not np.iscomplexobj(S) and not np.iscomplexobj(self.Khat):
            result = result.real
        return result
