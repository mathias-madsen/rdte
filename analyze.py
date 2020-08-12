import numpy as np
from matplotlib import pyplot as plt

import read
import approximation
import rotations


idx = 0

names = [
    "loopCW45-0to360_CC90d-360to0",  # initial Euler [-3.1416  0.      0.    ]
    "YloopCW45_CC90dTO360",  # initial Euler [-1.5708  0.      3.1416]
    "Y_45DegX_loopCW45_CC90dTO360"  # initial Euler [-0.7854  0.      3.1416]
    ]

name = names[idx]
path = "/Users/mathias/Downloads/%s.csv" % name

print("Path:")
print(path)
print()

with open(path, "r") as source:
    data = read.text2dict(source.read())

poses = read.extract_tcp_poses(data)
pose_xyz, pose_rvecs = np.split(poses, 2, axis=1)
pose_quats = rotations.rvecs2quats(pose_rvecs)
pose_rmats = np.stack([q.rotation_matrix for q in pose_quats])
pose_euler = rotations.matrix2euler(pose_rmats)
rel_pose_quats = [pose_quats[0].inverse * q for q in pose_quats]
# pose_euler = matrix2euler(pose_rmats)
rel_pose_rvecs = rotations.quats2rvecs(rel_pose_quats)
rel_pose_rmats = np.stack([q.rotation_matrix for q in rel_pose_quats])
rel_pose_euler = rotations.matrix2euler(rel_pose_rmats)

print("Initial Euler pose:")
print(pose_euler[0].round(4))
print()

forces = read.extract_tcp_forces(data)
force_xyz, force_rvec = np.split(forces, 2, axis=1)
rel_force_xyz = force_xyz @ pose_rmats[0].T
force_quats = rotations.rvecs2quats(force_rvec)
force_degrees = np.stack([q.degrees for q in force_quats])

plt.figure(figsize=(8, 3))
lines = plt.plot(rel_force_xyz, "-", alpha=0.5)
plt.legend(lines, list("xyz"))
plt.xlabel("Time step", fontsize=12)
plt.ylabel("Force", fontsize=12)
plt.tight_layout()
plt.savefig("/Users/mathias/Documents/mupsi/plots/forcexyz%s.pdf" % idx)
plt.close("all")

plt.figure(figsize=(8, 3))
radians = rel_pose_euler[:, 2] + pose_euler[0, 2]
degrees = (180/np.pi * radians) % 360
shift = 0
plt.plot(np.roll(degrees, shift),
         np.roll(rel_force_xyz[:, 0], shift), ".", alpha=0.5, label="x")
plt.plot(np.roll(degrees, shift),
         np.roll(rel_force_xyz[:, 1], shift), ".", alpha=0.5, label="y")
plt.plot(np.roll(degrees, shift),
         np.roll(rel_force_xyz[:, 2], shift), ".", alpha=0.5, label="z")
plt.xlabel("Angle (degrees)", fontsize=12)
plt.ylabel("Force", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/mathias/Documents/mupsi/plots/angleforce%s.pdf" % idx)
plt.close("all")

plt.figure(figsize=(12, 4))
lines = plt.plot(force_degrees, "-", alpha=0.5)
plt.title("Relative rotational force magnitudes (degrees)")
plt.tight_layout()
plt.savefig("/Users/mathias/Documents/mupsi/plots/forcerot%s.pdf" % idx)
plt.close("all")

speeds = read.extract_tcp_speed(data)
speed_xyz, speed_rvec = np.split(speeds, 2, axis=1)
# speed_rmats = rvecs2matrices(speed_rvec)
# speed_euler = matrix2euler(speed_rmats)

print("Initial direction of tool:")
print(pose_rmats[0, :, 2].round(4))
print()

rotmean = np.mean(rel_pose_rmats[:, :, :], axis=0)
rotvar = np.var(rel_pose_rmats[:, :, :], axis=0)
assert np.allclose(rotvar[:, 2], 0)  # z axis does not move
assert np.allclose(rotvar[2, 0], 0)  # z coordinate remain unchanged
assert np.allclose(rotmean[:2, 2], 0, atol=1e-3)
assert np.allclose(rotmean[2, :2], 0, atol=1e-3)

calm = np.all(speeds < 1e-3, axis=1)
calm_euler = rel_pose_euler[calm, :]
calm_forces = rel_force_xyz[calm, :]

old_x = calm_euler[:, 2]
old_y = calm_forces[:, :]
bandwidth = approximation.find_good_bandwidth(old_x, old_y)
new_x = np.linspace(-np.pi, +np.pi, 300)
new_y = approximation.resample(old_x, old_y, new_x, bandwidth=bandwidth)
smooth_y = np.transpose([approximation.low_pass_filter(y) for y in new_y.T])

figure, axlist = plt.subplots(nrows=3, figsize=(8, 9))
for i, axname in enumerate("xyz"):
    new_y_smooth = approximation.low_pass_filter(new_y[:, i])
    axlist[i].plot(old_x, old_y[:, i], ".", alpha=0.3)
    axlist[i].plot(new_x, smooth_y[:, i], "r-", lw=3, alpha=0.7)
    axlist[i].set_xlabel("Angle", fontsize=12)
    axlist[i].set_ylabel("Force $%s$" % axname, fontsize=12)
plt.tight_layout()
plt.savefig("/Users/mathias/Documents/mupsi/plots/regression%s.pdf" % idx)
plt.close("all")
