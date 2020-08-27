import numpy as np
from scipy.fftpack import rfft, irfft


def low_pass_filter(y, cutoff=10):
    """ Remove high-frequency oscillations from a time series.

    Parameters:
    -----------
    y : array of shape (N,)
        A timeseries of equally-spaced, real-valued floats.
    cutoff : int >= 0
        The index of the first oscillation frequency we choose to
        disregard because we consider it too high-frequent. All higher
        frequencies are disregarded too, all lower frequencies kept.

    Returns:
    yhat : array of shape (N,)
        A smoothed version of the input timeseries from which of the
        high-frequency components have been removed.
    """

    freqs = rfft(y)
    freqs[cutoff:] *= 0

    return irfft(freqs)


def get_windows(arr, width=20):
    """ Stack up time slices from a time series.

    Parameters:
    -----------
    arr : numpy array of shape (N, ...)
        An array whose first axis can be interpreted as a time index.
    width : int > 0
        The width of the desired windows.

    Returns:
    --------
    windows : numpy array of shape (N - width + 1, width, ...)
        A stack whose t'th entry is `arr[t : t + width, ...]`.
    """

    return np.array([arr[t:t + width] for t in range(len(arr) - width + 1)])


def compute_windowed_fourier(arr, width=20):
    """ Perform a Fourier analysis on each short time slice.

    Parameters:
    -----------
    arr : numpy array of shape (N, D)
        An array whose first axis can be interpreted as a time index.
    width : int > 0
        The width of the time slices that will be analyzed separately.

    Returns:
    --------
    freqs : numpy array of shape (N - width + 1, width, D)
        A representation of the frequency spectrum of each time slice,
        with `freqs[t, k, d]` being the Fourier coefficient for the
        frequency of `width / k` in slice number `t` and dimension `d`.
    """

    return rfft(get_windows(arr, width), axis=1)


def real_slow_fourier_transform(series):
    """ For testing, compute the `scipy.fftpack.rfft` explicitly. """

    period = 2 * np.pi * np.arange(len(series)) / len(series)
    freqs = np.zeros_like(series)
    freqs[0] = np.sum(series)

    for k, _ in enumerate(series[1::2]):
        wave = np.cos((k + 1) * period)
        freqs[1 + 2*k] = np.dot(wave, series)

    for k, _ in enumerate(series[2::2]):
        wave = -np.sin((k + 1) * period)
        freqs[2 + 2*k] = np.dot(wave, series)

    return freqs


def unevenly_spaced_discrete_real_fourier_series(x, y, num_freqs=None):
    """ Decompose a function into periodic signals.

    Coincides with Discrete, real-valued Fourier analysis when `x`
    is evenly spaced on [0, 2*pi).
    """

    assert x.shape == y.shape
    assert x.ndim == y.ndim == 1

    width = len(x)
    height = len(x) if num_freqs is None else num_freqs

    waves = np.ones([height, width], dtype=x.dtype)
    for k, _ in enumerate(waves[1::2, :]):
        waves[1 + 2*k] = np.cos((k + 1) * x)
    for k, _ in enumerate(waves[2::2, :]):
        waves[2 + 2*k] = -np.sin((k + 1) * x)

    short_freqs = np.sum(waves * y, axis=1)

    long_freqs = np.zeros(len(x))
    long_freqs[:num_freqs] = short_freqs

    return long_freqs
