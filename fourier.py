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



def one(k, n):
    a = np.zeros(n)
    a[k] = 1
    return a


def slow_irfft(freqs, x=None):
    """ Explicitly compute irfft, for testing purposes. """

    freqs = np.float64(np.copy(freqs))

    assert np.ndim(freqs) == 1
    N = len(freqs)
    assert N > 2, ("Please provide 2 or more freqs, not %s" % N)
    constant = freqs[0] / len(freqs)

    if x is None:
        x = 2 * np.pi * np.arange(N) / N

    coscoeffs = freqs[1::2]
    cosfreqs = 1 + np.arange(len(coscoeffs))
    cost = 2.0 * np.cos(x[:, None] * cosfreqs)
    assert cost.ndim == 2
    assert cost.shape[1] == len(cosfreqs)
    if N % 2 == 0:
        coscoeffs[-1] *= 0.5
    cosine = np.sum(coscoeffs * cost, axis=1)
    cosine *= 1 / N

    sinecoeffs = freqs[2::2]
    sinfreqs = 1 + np.arange(len(sinecoeffs))
    sint = 2.0 * np.sin(x[:, None] * sinfreqs)
    sine = np.sum(sinecoeffs * sint, axis=1)
    sine *= 1 / N

    return constant + cosine - sine


def get_first_index_of_tail_of_zeros(series, atol=1e-5):
    """ Get the point at which a series switches to all zeros.
    
    Notes:
    ------
    Returns the length of the series if it contains no zeros.

    Examples:
    ---------
    >>> get_first_index_of_tail_of_zeros([1, 1, 1, 0, 0, 0, 0])
    3
    >>> get_first_index_of_tail_of_zeros([0, 0, 1, 0, 0, 0, 0])
    3
    >>> get_first_index_of_tail_of_zeros([1, 1, 1, 1, 1, 1, 0])
    6
    >>> get_first_index_of_tail_of_zeros([1, 1, 1, 1, 1, 1, 1])
    7
    """

    assert np.ndim(series) == 1

    is_zero = np.isclose(series, 0, atol=atol)

    if not np.any(is_zero):
        return len(series)

    zeros_from_here = np.cumprod(is_zero[::-1], dtype=bool)[::-1]

    # we rely on the fact that `np.argmax` returns the index of the
    # _first_ occurrence of the maximum value in case there is more
    # than one in the array. In this case, the maximum is `True`,
    # so `np.argmax` returns the index of the first True element.
    return np.argmax(zeros_from_here)


class InverseFourierFunction:

    def __init__(self, freqs):

        assert np.ndim(freqs) == 1
        assert len(freqs) > 2, ("Need >2 or more freq, got %s" % len(freqs))

        self.freqs = np.float64(np.copy(freqs))
        self.N = len(self.freqs)
        self.constant = self.freqs[0] / self.N

        self.coscoeffs = self.freqs[1::2]
        self.coscoeffs *= 2.0 / self.N
        if self.N % 2 == 0:
            self.coscoeffs[-1] *= 0.5
        self.cosfreqs = 1 + np.arange(len(self.coscoeffs))

        self.sincoeffs = self.freqs[2::2]
        self.sincoeffs *= -2 / self.N
        self.sinfreqs = 1 + np.arange(len(self.sincoeffs))

        # slice off nearly-zero tails of the frequency table:
        cos_truncation = get_first_index_of_tail_of_zeros(self.coscoeffs)
        sin_truncation = get_first_index_of_tail_of_zeros(self.sincoeffs)
        idx = max(cos_truncation, sin_truncation)        
        self.coscoeffs = self.coscoeffs[:idx]
        self.cosfreqs = self.cosfreqs[:idx]
        self.sincoeffs = self.sincoeffs[:idx]
        self.sinfreqs = self.sinfreqs[:idx]

    def __call__(self, x):

        if np.isscalar(x):
            x = np.atleast_1d(x)

        assert np.ndim(x) == 1

        cos_waves = np.cos(x[:, None] * self.cosfreqs)
        cos = np.sum(self.coscoeffs * cos_waves, axis=1)

        sin_waves = np.sin(x[:, None] * self.sinfreqs)
        sin = np.sum(self.sincoeffs * sin_waves, axis=1)

        return self.constant + cos + sin
    
    def export_as_code(self):

        terms = ["%.5g" % self.constant]

        for coeff, freq in zip(self.coscoeffs, self.cosfreqs):
            terms.append("%.5g*np.cos(%.5g*x)" % (coeff, freq))

        for coeff, freq in zip(self.sincoeffs, self.sinfreqs):
            terms.append("%.5g*np.sin(%.5g*x)" % (coeff, freq))
        
        return " + ".join(terms)


def _test_slow_irfft_constants():

    assert np.allclose(irfft([1]), [1/1])
    assert np.allclose(irfft([1, 0]), [1/2, 1/2])
    assert np.allclose(irfft([1, 0, 0]), [1/3, 1/3, 1/3])
    assert np.allclose(irfft([1, 0, 0, 0]), [1/4, 1/4, 1/4, 1/4])
    assert np.allclose(irfft([1, 0, 0, 0, 0]), [1/5, 1/5, 1/5, 1/5, 1/5])

    # assert np.allclose(slow_irfft([1]), [1/1])
    # assert np.allclose(slow_irfft([1, 0]), [1/2, 1/2])
    assert np.allclose(slow_irfft([1, 0, 0]), [1/3, 1/3, 1/3])
    assert np.allclose(slow_irfft([1, 0, 0, 0]), [1/4, 1/4, 1/4, 1/4])
    assert np.allclose(slow_irfft([1, 0, 0, 0, 0]), [1/5, 1/5, 1/5, 1/5, 1/5])


def _test_slow_irfft_additivity():

    for k in range(10):
        freqs = np.random.normal(size=k + 3)
        simultaneously = irfft(freqs)
        piecewise = np.sum([irfft(f) for f in np.diag(freqs)], axis=0)
        assert np.allclose(simultaneously, piecewise)

    for k in range(10):
        freqs = np.random.normal(size=k + 3)
        simultaneously = slow_irfft(freqs)
        piecewise = np.sum([slow_irfft(f) for f in np.diag(freqs)], axis=0)
        assert np.allclose(simultaneously, piecewise)


def _test_slow_irfft_first_cosine():

    tau = 2 * np.pi
    wave = lambda k: 2/k * np.cos(1 * tau * np.arange(k) / k)

    assert np.allclose(irfft([0, 1, 0]), wave(3))
    assert np.allclose(irfft([0, 1, 0, 0]), wave(4))
    assert np.allclose(irfft([0, 1, 0, 0, 0]), wave(5))
    assert np.allclose(irfft([0, 1, 0, 0, 0, 0]), wave(6))

    assert np.allclose(slow_irfft([0, 1, 0]), wave(3))
    assert np.allclose(slow_irfft([0, 1, 0, 0]), wave(4))
    assert np.allclose(slow_irfft([0, 1, 0, 0, 0]), wave(5))
    assert np.allclose(slow_irfft([0, 1, 0, 0, 0, 0]), wave(6))


def _test_slow_irfft_first_sine():

    tau = 2 * np.pi
    wave = lambda k: -2/k * np.sin(1 * tau * np.arange(k) / k)

    assert np.allclose(irfft([0, 0, 1]), wave(3))
    assert np.allclose(irfft([0, 0, 1, 0]), wave(4))
    assert np.allclose(irfft([0, 0, 1, 0, 0]), wave(5))
    assert np.allclose(irfft([0, 0, 1, 0, 0, 0]), wave(6))

    assert np.allclose(slow_irfft([0, 0, 1]), wave(3))
    assert np.allclose(slow_irfft([0, 0, 1, 0]), wave(4))
    assert np.allclose(slow_irfft([0, 0, 1, 0, 0]), wave(5))
    assert np.allclose(slow_irfft([0, 0, 1, 0, 0, 0]), wave(6))


def _test_slow_irfft_second_cosine():

    tau = 2 * np.pi
    wave = lambda k: 2/k * np.cos(2 * tau * np.arange(k) / k)

    assert np.allclose(irfft([0, 0, 0, 1, 0]), wave(5))
    assert np.allclose(irfft([0, 0, 0, 1, 0, 0]), wave(6))
    assert np.allclose(irfft([0, 0, 0, 1, 0, 0, 0]), wave(7))
    assert np.allclose(irfft([0, 0, 0, 1, 0, 0, 0, 0]), wave(8))

    assert np.allclose(slow_irfft([0, 0, 0, 1, 0]), wave(5))
    assert np.allclose(slow_irfft([0, 0, 0, 1, 0, 0]), wave(6))
    assert np.allclose(slow_irfft([0, 0, 0, 1, 0, 0, 0]), wave(7))
    assert np.allclose(slow_irfft([0, 0, 0, 1, 0, 0, 0, 0]), wave(8))

    # special case when it's the last, and N is even:
    assert np.allclose(irfft([0, 0, 0, 1]), 0.5 * wave(4))
    assert np.allclose(slow_irfft([0, 0, 0, 1]), 0.5 * wave(4))


def _test_slow_irfft_one_onehot_vectors():

    for n in range(3, 10):
        for k in range(n):
            freqs = one(k, n)
            theirs = irfft(freqs)
            ours = slow_irfft(freqs)
            assert np.allclose(theirs, ours)


def _test_slow_irfft_with_random_inputs():

    for freqs in np.random.normal(size=(10, 4)):
        theirs = irfft(freqs)
        ours = slow_irfft(freqs)
        assert np.allclose(theirs, ours, atol=1e-5)

    for freqs in np.random.normal(size=(10, 5)):
        theirs = irfft(freqs)
        ours = slow_irfft(freqs)
        assert np.allclose(theirs, ours, atol=1e-5)

    for freqs in np.random.normal(size=(10, 50)):
        theirs = irfft(freqs)
        ours = slow_irfft(freqs)
        assert np.allclose(theirs, ours, atol=1e-5)

    for freqs in np.random.normal(size=(10, 51)):
        theirs = irfft(freqs)
        ours = slow_irfft(freqs)
        assert np.allclose(theirs, ours, atol=1e-5)


def _test_inverse_dft_function():

    for N in range(3, 15):
        freqs = np.random.normal(size=N)
        default_x = 2 * np.pi * np.arange(N) / N
        other_x = np.random.uniform(0, 2 * np.pi, size=N)
        inverse = InverseFourierFunction(freqs)
        assert np.allclose(inverse(default_x), irfft(freqs))
        assert np.allclose(inverse(default_x), slow_irfft(freqs))
        assert np.allclose(inverse(other_x), slow_irfft(freqs, x=other_x))
        print(inverse.export_as_code())
        print()


def demo_slow_irfft():

    from matplotlib import pyplot as plt

    x = 2 * np.pi * np.arange(100) / 100
    y = np.cumsum(np.random.normal(size=100))

    print("Lowpass")
    freqs = rfft(y)
    freqs[20:] *= 0
    yhat = irfft(freqs)
    print("Done.\n")

    print("Eval")    
    xthin = 2 * np.pi * np.arange(30) / 30
    ythin = slow_irfft(freqs, xthin)
    print("Done.\n")    

    plt.plot(x, y, ".")
    plt.plot(x, yhat, "-", alpha=0.3)
    plt.plot(xthin, ythin, "-", alpha=0.3)
    plt.show()


if __name__ == "__main__":

    _test_slow_irfft_constants()
    _test_slow_irfft_additivity()
    _test_slow_irfft_first_cosine()
    _test_slow_irfft_first_sine()
    _test_slow_irfft_second_cosine()
    _test_slow_irfft_one_onehot_vectors()
    _test_slow_irfft_with_random_inputs()
    _test_inverse_dft_function()

    print("Fourier module passed all tests.\n")
