import numpy as np

# methods exported from scipy, to avoid the dependency
# https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/windows/windows.py

def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1

def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False

def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w

def general_cosine(M, a, sym=True):
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)

    return _truncate(w, needs_trunc)


def grid(image_shape: tuple):
    return
