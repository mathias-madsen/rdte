import tqdm
import numpy as np
from scipy.fftpack import rfft, irfft


def resample(old_x, old_y, new_x, bandwidth=None):
    """ Use Gaussian smoothing to interpolate a function.

    Parameters:
    -----------
    old_x : array of shape (N,)
        An array of input values at which the function value is known.
    old_y : array of shape (N, ...)
        The value of the function at the points given by `old_x`.
    new_x : array of shape (K,)
        An array of locations at which we wish to evaluate the function.
    bandwidth : float >= 0, optional
        A parameter that controls how large the neighborhood of nearby
        values that will be taken into account during the interpolation.
        When `bandwidth ~= 0`, this method behaves approximately like
        nearest-neighbor interpolation. When `bandwidth --> inf`, it
        behaves approximately like a constant-baseline prediction. The
        unit of the bandwidth is `1 / u^2`, where `u` is the unit of `x`.

    Returns:
    --------
    new_y : array of shape (K, ...)
        The interpolated values of the function at the input locations
        given by `new_x`.
    """

    if bandwidth is None:
        bandwidth = 1 / np.mean(np.diff(np.sort(old_y)) ** 2)

    old_x_2d, new_x_2d = np.meshgrid(old_x, new_x)
    deviations = bandwidth * (old_x_2d - new_x_2d) ** 2
    deviations -= np.min(deviations, axis=1, keepdims=True)
    weights = np.exp(-deviations)
    weights /= weights.sum(axis=1, keepdims=True)

    # we perform the multiplication (K, N) * (1, N, ...) by
    # converting it into (N, K) * (..., N, 1), letting numpy's
    # broadcasting rules do the logwork, and then converting
    # the resulting array back to shape (K, N, ...):
    weighted = (weights.T * old_y[None].T).T

    # we can then average out the N axis, leaving K values:
    return np.sum(weighted, axis=1)


def find_good_bandwidth(x, y, resolution=40, logfactor=10):
    """ Use cross-validation to select a promising bandwidth. """

    # first compute a decent suggestion in a deterministic fashion:
    suggested = 1. / np.mean(np.diff(np.sort(x)) ** 2)

    # then collect a series of variants of that suggestion::
    scalings = np.logspace(-logfactor, +logfactor, resolution)
    candidates = scalings * suggested

    # split the data set up into two subsets:
    is_train = np.random.binomial(n=1, p=0.9, size=len(x)).astype(bool)
    x_train = x[is_train]
    y_train = y[is_train]
    x_val = x[~is_train]
    y_val = y[~is_train]

    # collect the errors associated with each candidate bandwidth:
    print("Searching for best smoothing bandwidth . . . ")
    errors = []
    for bandwidth in tqdm.tqdm(candidates):
        y_val_hat = resample(x_train, y_train, x_val, bandwidth)
        error = np.mean((y_val - y_val_hat) ** 2)
        errors.append(error)

    winning_idx = np.argmin(errors)
    winning_val = candidates[winning_idx]

    print("Initial value: exp(%.5f)" % np.log(suggested))
    print("Winning value: exp(%.5f)" % np.log(winning_val))
    print()

    return winning_val


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
