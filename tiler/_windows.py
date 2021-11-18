# Heavily based on scipy code
# For more information please refer to scipy/signal/windows/windows.py
from typing import Union, List
import numpy as np


def _len_guards(m: int):
    """Handle small or incorrect window lengths"""
    if int(m) != m or m < 0:
        raise ValueError("Window length m must be a non-negative integer")
    return m <= 1


def _general_cosine(m: int, a: Union[np.ndarray, List]):
    """Generic weighted sum of cosine terms window"""
    if _len_guards(m):
        return np.ones(m)

    fac = np.linspace(-np.pi, np.pi, m)
    w = np.zeros(m)
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)

    return w


def _general_hamming(m: int, alpha: float):
    """Return a generalized Hamming window."""
    return _general_cosine(m, [alpha, 1.0 - alpha])


def _boxcar(m: int):
    """Return a boxcar or rectangular window.

    Also known as a rectangular window or Dirichlet window, this is equivalent
    to no window at all: all weights are equal to one.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: boxcar window
    """
    if _len_guards(m):
        return np.ones(m)
    return np.ones(m, dtype=float)


def _triang(m: int):
    """Return a triangular window.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: triangular window
    """
    if _len_guards(m):
        return np.ones(m)

    n = np.arange(1, (m + 1) // 2 + 1)
    if m % 2 == 0:
        w = (2 * n - 1.0) / m
        w = np.r_[w, w[::-1]]
    else:
        w = 2 * n / (m + 1.0)
        w = np.r_[w, w[-2::-1]]
    return w


def _parzen(m: int):
    """Return a Parzen window.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: Parzen window
    """
    if _len_guards(m):
        return np.ones(m)

    n = np.arange(-(m - 1) / 2.0, (m - 1) / 2.0 + 0.5, 1.0)
    na = np.extract(n < -(m - 1) / 4.0, n)
    nb = np.extract(abs(n) <= (m - 1) / 4.0, n)
    wa = 2 * (1 - np.abs(na) / (m / 2.0)) ** 3.0
    wb = 1 - 6 * (np.abs(nb) / (m / 2.0)) ** 2.0 + 6 * (np.abs(nb) / (m / 2.0)) ** 3.0
    w = np.r_[wa, wb, wa[::-1]]
    return w


def _bohman(m: int):
    """Return a Bohman window.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: Bohman window
    """
    if _len_guards(m):
        return np.ones(m)

    fac = np.abs(np.linspace(-1, 1, m)[1:-1])
    w = (1 - fac) * np.cos(np.pi * fac) + 1.0 / np.pi * np.sin(np.pi * fac)
    w = np.r_[0, w, 0]
    return w


def _blackman(m: int):
    """Return a minimum 4-term Blackman-Harris window according to Nuttall.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: minimum 4-term Blackman-Harris window according to Nuttall
    """
    return _general_cosine(m, [0.42, 0.50, 0.08])


def _nuttall(m: int):
    """Return a minimum 4-term Blackman-Harris window according to Nuttall.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: minimum 4-term Blackman-Harris window
    """
    return _general_cosine(m, [0.3635819, 0.4891775, 0.1365995, 0.0106411])


def _blackmanharris(m: int):
    """Return a minimum 4-term Blackman-Harris window.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: minimum 4-term Blackman-Harris window
    """
    return _general_cosine(m, [0.35875, 0.48829, 0.14128, 0.01168])


def _bartlett(m: int):
    """Return a Bartlett window.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: Bartlett window
    """
    if _len_guards(m):
        return np.ones(m)

    n = np.arange(0, m)
    w = np.where(
        np.less_equal(n, (m - 1) / 2.0), 2.0 * n / (m - 1), 2.0 - 2.0 * n / (m - 1)
    )

    return w


def _hann(m: int):
    """Return a Hann window.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: Hann window
    """
    return _general_hamming(m, 0.5)


def _barthann(m: int):
    """Return a modified Bartlett-Hann window.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: modified Bartlett-Hann window
    """
    if _len_guards(m):
        return np.ones(m)

    n = np.arange(0, m)
    fac = np.abs(n / (m - 1.0) - 0.5)
    w = 0.62 - 0.48 * fac + 0.38 * np.cos(2 * np.pi * fac)

    return w


def _hamming(m: int):
    """Return a Hamming window.

    Args:
        m (int): Number of points in the output window. If zero or less, an empty array is returned.

    Returns:
        np.ndarray: Hamming window
    """
    return _general_hamming(m, 0.54)


_mapping = {
    "boxcar": _boxcar,
    "triang": _triang,
    "parzen": _parzen,
    "bohman": _bohman,
    "blackman": _blackman,
    "nuttall": _nuttall,
    "blackmanharris": _blackmanharris,
    "bartlett": _bartlett,
    "hann": _hann,
    "barthann": _barthann,
    "hamming": _hamming,
}


def get_window(window: str, length: int):
    fn = _mapping.get(window, None)
    if not fn:
        raise ValueError(f"Window function {window} is not supported.")

    return fn(length)
