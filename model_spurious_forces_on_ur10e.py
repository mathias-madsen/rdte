import os

import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from scipy.fftpack import rfft, irfft

import read
import fourier
import smoothing
import approximation


# filepaths = [
#     "data/runedata/ur10_parallel_1",
#     "data/runedata/ur10_perpendicular_1",
#     "data/runedata/ur10_random_1",
# ]

filepaths = [
    "data/ur10e_no_guiding_3/loopCCW90dTO360.csv",
    "data/ur10e_no_guiding_3/loopCW45-0to360_CC90d-360to0.csv",
    "data/ur10e_no_guiding_3/loopCW45dTO360.csv",
]

records = OrderedDict()
reclist = []
for fp in filepaths:
    rec = read.Recording(fp)
    records[fp] = rec
    reclist.append(rec)


for k in range(6):
    figure, axlist = plt.subplots(nrows=3, figsize=(8, 6), sharex=True)
    allx = []
    ally = []
    for i, (filename, rec) in enumerate(records.items()):
        x = rec.joint_radians[:, 5] % (2 * np.pi)
        y = rec.force[:, k]
        allx.append(x)
        ally.append(y)
        newx = 2 * np.pi * np.arange(100) / 100
        newy = approximation.resample(x, y, newx, bandwidth=1000.0)
        freqs = fourier.rfft(newy)
        freqs[10:] *= 0
        function = fourier.InverseFourierFunction(freqs)
        axlist[i].plot(x, y, ".", alpha=0.3, label="data")
        # axlist[i].plot(newx, newy, "r-", alpha=0.5, lw=4, label="smooth")
        axlist[i].plot(newx, function(newx), "k-", alpha=0.5, lw=4, label="function")
        axlist[i].set_title(filename)
        axlist[i].set_ylabel("force[%s]" % k)
    allx = np.concatenate(allx, axis=0)
    ally = np.concatenate(ally, axis=0)
    newx = 2 * np.pi * np.arange(100) / 100
    newy = approximation.resample(allx, ally, newx, bandwidth=1000.0)
    yhat = fourier.low_pass_filter(newy)
    freqs = fourier.rfft(newy)
    freqs[10:] *= 0
    function = fourier.InverseFourierFunction(freqs)
    print("force axis %s:" % k)
    print(function.export_as_code())
    print()
    axlist[-1].set_xlabel("joint_radians[5]")
    axlist[-1].legend()
    plt.savefig("plots/smooth/ur10_smoothened_axis_%s.png" % k)
    plt.show()


