import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import rotations
from read import Recording
from neural import MultiLayerPerceptron
from neural import MeanSquaredErrorMinimizer


dirpath = "data/runedata/"
filenames = [fn for fn in os.listdir(dirpath) if fn.startswith("ur5")]
filepaths = [os.path.join(dirpath, fn) for fn in filenames]


def get_radians(rmats):
    """ Get the angles of rotation around z relative to first frame. """

    relmats = rmats[0].T @ rmats
    xs, ys = relmats[:, :2, 0].T

    # verify the assumption that we are turning around the z axis:
    meanmat = np.mean(relmats, axis=0)
    stdmat = np.std(relmats, axis=0)
    assert np.allclose(meanmat[:, 2], [0, 0, 1], atol=1e-4)
    assert np.allclose(stdmat[:, 2], [0, 0, 0], atol=1e-4)

    return np.arctan2(ys, xs)


print("Collecting data . . .", end=" ", flush=True)
rmats = []
fxyzs = []
for filename in filenames:
    rec = Recording("data/runedata/%s" % filename)
    rmats.append(rec.pose_rmats)
    fxyzs.append(rec.force_xyz)
    print(".", end=" ", flush=True)
x = np.concatenate(rmats, axis=0).reshape([-1, 9])
y = np.concatenate(fxyzs, axis=0).reshape([-1, 3])
print("Done.\n")


network = MultiLayerPerceptron(9, 3)
trainer = MeanSquaredErrorMinimizer(network)

session = trainer.session
session.run(tf.global_variables_initializer())

print("TRAINING:")
print("---------")

lrates = np.linspace(0.1, 0.0, 3000)
for iteration, lrate in enumerate(lrates):
    trainer.step(x, y, lrate)
    if (iteration + 1) % 10 == 0:
        feed = {trainer.X: x, trainer.Y: y}
        loss = trainer.session.run(trainer.LOSS, feed)
        print(loss)


yhat = trainer.session.run(trainer.YHAT, {trainer.X: x})
figure, axlist = plt.subplots(nrows=3)
for i, color in enumerate("rgb"):
    axlist[i].plot(y[:, i], ".", alpha=0.1, color=color)
    axlist[i].plot(yhat[:, i], "-", lw=3, alpha=1.0, color="black")
plt.tight_layout()
plt.show()