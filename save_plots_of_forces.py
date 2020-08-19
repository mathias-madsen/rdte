import os
import numpy as np
from matplotlib import pyplot as plt

import rotations
from read import Recording


dirpath = "data/runedata/"
filenames = [fn for fn in os.listdir(dirpath) if fn.startswith("ur")]
filepaths = [os.path.join(dirpath, fn) for fn in filenames]
# filepaths = [fp for fp in filepaths if "random" not in fp]


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


for filename in filenames:

    print(filename, ". . . ")

    filepath = "data/runedata/%s" % filename
    rec = Recording(filepath)

    # dz = get_radians(rec.pose_rmats)
    # dz = (dz + 2*np.pi) % (2*np.pi)

    dz = rec.joint_radians[:, -1]

    fxyz = rec.force_xyz
    # fxyz = np.squeeze(rec.pose_rmats.transpose([0, 2, 1]) @ fxyz[:, :, None])
    fxyz = np.squeeze(rec.pose_rmats.transpose([0, 1, 2]) @ fxyz[:, :, None])
    fx, fy, fz = fxyz.T

    figure, axlist = plt.subplots(figsize=(12, 8), nrows=3, sharex=True)
    axlist[0].plot(dz, fx, "r.", alpha=0.5)
    axlist[1].plot(dz, fy, "g.", alpha=0.5)
    axlist[2].plot(dz, fz, "b.", alpha=0.5)
    axlist[0].set_ylabel("force x")
    axlist[1].set_ylabel("force y")
    axlist[2].set_ylabel("force z")
    axlist[2].set_xlabel("Angle of rotation around z (radians)")
    plt.suptitle(filename)
    plt.tight_layout()
    plt.savefig("plots/%s_forces_xyz_over_angles.png" % filename)
    plt.close("all")

    figure, axlist = plt.subplots(figsize=(12, 10), nrows=4, sharex=True)
    axlist[0].plot(fx, "r.", alpha=0.5)
    axlist[1].plot(fy, "g.", alpha=0.5)
    axlist[2].plot(fz, "b.", alpha=0.5)
    axlist[3].plot(dz, "k.", alpha=0.5)
    axlist[0].set_ylabel("force x")
    axlist[1].set_ylabel("force y")
    axlist[2].set_ylabel("force z")
    axlist[3].set_ylabel("Angle (degrees)")
    axlist[3].set_xlabel("Time step (1/125s)")
    plt.suptitle(filename)
    plt.tight_layout()
    plt.savefig("plots/%s_forces_xyz_over_time.png" % filename)
    plt.close("all")


# figure, axlist = plt.subplots(figsize=(6, 10), nrows=5)

# for fp, axes in zip(filepaths, axlist):

#     print("Loading %r . . ." % fp)
#     rec = Recording(fp)
#     print("Done.\n")

#     vecx = rec.pose_rmats[:, :, 0]
#     vecy = rec.pose_rmats[:, :, 1]
#     vecz = rec.pose_rmats[:, :, 2]

#     sx, sy, sz = np.std(vecz, axis=0)
    
#     print("Variance of nose direction:")
#     print("[%.5f, %.5f, %.5f]." % (sx, sy, sz))
#     print("")

#     axes.plot(rec.force_xyz, ".-", alpha=0.5)
#     axes.set_title(fp)

#     # euler = rotations.matrix2euler(rec.pose_rmats)
#     # dz = 180/np.pi * euler[:, 2]
#     # fx, fy, fz = rec.force_xyz.T
#     # plt.plot(dz, fx, ".", alpha=0.1, label="x")
#     # plt.plot(dz, fy, ".", alpha=0.1, label="y")
#     # plt.plot(dz, fz, ".", alpha=0.1, label="z")
#     # plt.legend()
#     # plt.show()

# plt.show()
