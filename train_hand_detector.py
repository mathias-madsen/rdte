import numpy as np
from matplotlib import pyplot as plt

from read import Recording
from fourier import compute_windowed_fourier

import logistic


NAMES = [
    "3d rotation",
    "z rotation",
    "no rotation",
    ]

BOUNDS = [
    (200, 1530),
    (160, 1800),
    (270, 1570),
    ]


def extract_data(width=20):

    insides = []
    outsides = []

    for name, (start, stop) in zip(NAMES, BOUNDS):

        rec = Recording("data/ur10e_guiding/%s.csv" % name)

        # negative data from before touch
        head = rec.force_xyz[:start]
        head_freqs = compute_windowed_fourier(head, width=width)
        outsides.append(head_freqs)

        # negative data from after touch
        hand = rec.force_xyz[start + 100:stop - 100]
        hand_freqs = compute_windowed_fourier(hand, width=width)
        insides.append(hand_freqs)

        # positive data from middle section
        tail = rec.force_xyz[stop:]
        tail_freqs = compute_windowed_fourier(tail, width=width)
        outsides.append(tail_freqs)
    
    insides = np.concatenate(insides, axis=0)
    outsides = np.concatenate(outsides, axis=0)

    return insides, outsides


def preprocess(data, degree=2):
    """ Convert data to feature vectors. """
    length = data.shape[0]
    flat = np.reshape(data, [length, -1])
    powers = [flat ** (k + 1) for k in range(degree)]
    return np.concatenate(powers, axis=1)


def plot_recording(name):

    rec = Recording("data/ur10e_guiding/%s.csv" % name)
    forces = rec.force_xyz
    freqs = compute_windowed_fourier(forces, width=20)

    figure, axlist = plt.subplots(figsize=(12, 8), nrows=4, sharex=True)
    figure.suptitle(name)
    for i, axes in enumerate(axlist[:3]):
        powers = freqs[:, :, 0].T ** 2
        powers /= powers.max(axis=0)
        axes.imshow(powers, aspect="auto")
    axlist[-1].plot(forces, ".-")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    width = 30
    snipoff = 6
    
    positive, negative = extract_data(width)

    # add squares as features:
    posfeats = preprocess(positive[:, :, :snipoff])
    negfeats = preprocess(negative[:, :, :snipoff])

    x = np.concatenate([posfeats, negfeats], axis=0)
    y = np.array(len(posfeats) * [True] + len(negfeats) * [False])

    idx = np.random.permutation(len(x))
    test_idx, train_idx = np.split(idx, [len(idx) // 10])

    train_x = x[train_idx,]
    train_y = y[train_idx,]

    test_x = x[test_idx,]
    test_y = y[test_idx,]

    solver = logistic.LogisticRegressionSolver(train_x, train_y, regweight=0.0)
    solution = solver.solve()

    print("Training set accuracy:")
    print(logistic.compute_accuracy(solution.x, train_x, train_y))
    print(logistic.compute_logistic_loss(solution.x, train_x, train_y))
    print()
    print("Validation set accuracy:")
    print(logistic.compute_accuracy(solution.x, test_x, test_y))
    print(logistic.compute_logistic_loss(solution.x, test_x, test_y))
    print()

    rec = Recording("data/ur10e_guiding/%s.csv" % NAMES[0])
    freqs = compute_windowed_fourier(rec.force_xyz, width=width)
    data = preprocess(freqs[:, :, :snipoff])
    probs = logistic.compute_probs(solution.x, data)

    figure, axlist = plt.subplots(figsize=(12, 8), nrows=4, sharex=True)
    for i, axes in enumerate(axlist[:3]):
        powers = freqs[:, :, 0].T ** 2
        powers /= powers.max(axis=0)
        axes.imshow(powers, aspect="auto")
    axlist[-1].plot(probs, ".-")
    plt.tight_layout()
    plt.show()
