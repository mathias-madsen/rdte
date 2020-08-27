import numpy as np


def gaussian(width):
    """ Return a Gaussian kernel of a given width. """

    span = np.linspace(-5, +5, width)
    weights = np.exp(-0.5 * span**2)
    weights /= weights.sum()

    return weights


def uniform(width):
    """ Return a vector of `width` identical terms summing to 1. """

    return np.ones(width) / width


def smoothen(series, width=101, kernel=None):
    """ Convolve a series or stack of series with a kernel. """

    if len(series) < width:
        raise ValueError("Too short: %s < %s" % (len(series), width))

    if kernel is None:
        kernel = gaussian(width)

    if series.ndim == 1:
        return np.convolve(kernel, series, mode="valid")
    elif series.ndim == 2:
        smoothT = [np.convolve(kernel, z, mode="valid") for z in series.T]
        return np.transpose(smoothT)
    else:
        raise ValueError("series.ndim must be 1 or 2, not %s" % series.ndim)


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    width = 25

    a = np.cumsum(np.random.normal(size=200))
    b = smoothen(a, width=width)
    
    aspan = np.arange(len(a))
    bspan = np.arange(len(b)) + width/2

    plt.plot(aspan, a, ".")
    plt.plot(bspan, b, "-", lw=3)
    plt.show()

    aa = np.cumsum(np.random.normal(size=(200, 3)), axis=0)
    bb = smoothen(aa, width=width)

    aspan = np.arange(len(a))
    bspan = np.arange(len(b)) + width/2

    plt.plot(aspan, aa[:, 0], "r.")
    plt.plot(bspan, bb[:, 0], "r-", lw=3)
    plt.plot(aspan, aa[:, 1], "b.")
    plt.plot(bspan, bb[:, 1], "b-", lw=3)
    plt.plot(aspan, aa[:, 2], "g.")
    plt.plot(bspan, bb[:, 2], "g-", lw=3)
    plt.show()