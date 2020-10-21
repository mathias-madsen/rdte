import numpy as np
from matplotlib import pyplot as plt

from read import Recording


np.set_printoptions(precision=5, suppress=True)

path1 = "data/should_be_the_same/ur10e_verticalDown_rotateby10_posDirection_fixedPose.csv"
path2 = "data/should_be_the_same/ur10e_verticalDown-pos2_rotateby10_posDirection_fixedPose.csv"
path3 = "data/should_be_the_same/ur10e_verticalDown-pos3_rotateby10_posDirection_fixedPose.csv"

figure, axes_list = plt.subplots(nrows=3, figsize=(8, 7))


for path, color in zip([path1, path2, path3], "rgb"):

    rec = Recording(path)

    print(rec.joint_radians.shape)
    print(rec.joint_radians.mean(axis=0))
    print(rec.joint_radians.std(axis=0))
    print()

    wrist_radians = rec.joint_radians[:, -1]
    for axes, force in zip(axes_list, rec.raw_force_xyz.T):
        axes.plot(wrist_radians, force, ".", alpha=0.1, color=color)



axes_list[0].set_ylabel("x-force")
axes_list[1].set_ylabel("y-force")
axes_list[2].set_ylabel("z-force")

axes_list[2].set_xlabel("wrist angle (radians)")

plt.tight_layout()
plt.show()