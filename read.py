import time
import numpy as np

import rotations


def text2dict(text):
    """ Convert a file string to a dict of equally-long arrays. """

    lines = text.strip().split("\n")
    headers = lines.pop(0).split(" ")
    data = {k: [] for k in headers}

    for line in lines:
        for h, c in zip(headers, line.split(" ")):
            data[h].append(float(c))

    return {k: np.array(v) for k, v in data.items()}


def extract_tcp_poses(data):
    """ Extract the shape-(N, 6) array of TCP poses. """

    names = ["actual_TCP_pose_%s" % i for i in range(6)]
    columns = [data[name] for name in names]

    return np.stack(columns, axis=1)


def extract_tcp_forces(data):
    """ Extract the shape-(N, 6) array of TCP poses. """

    names = ["actual_TCP_force_%s" % i for i in range(6)]
    columns = [data[name] for name in names]

    return np.stack(columns, axis=1)


def extract_wrench_forces(data):
    """ Extract the shape-(N, 6) array of raw wrench forces. """

    names = ["ft_raw_wrench_%s" % i for i in range(6)]
    columns = [data[name] for name in names]

    return np.stack(columns, axis=1)


def extract_raw_forces(data):
    """ Extract the shape-(N, 6) array of raw wrench forces. """

    names = ["ft_raw_wrench_%s" % i for i in range(6)]
    columns = [data[name] for name in names]

    return np.stack(columns, axis=1)


def extract_tcp_speed(data):
    """ Extract the shape-(N, 6) array of TCP speeds. """

    names = ["actual_TCP_speed_%s" % i for i in range(6)]
    columns = [data[name] for name in names]

    return np.stack(columns, axis=1)


def extract_joint_angles(data):
    """ Extract the shape-(N, 6) array of joint angles in radians. """

    names = ["actual_q_%s" % i for i in range(6)]
    columns = [data[name] for name in names]

    return np.stack(columns, axis=1)



class Recording:

    def __init__(self, csvpath):

        start = time.perf_counter()
        print("Reading record at %r . . ." % csvpath)

        with open(csvpath, "r") as source:
            self.data = text2dict(source.read())

        dur = time.perf_counter() - start
        print("Done after %.3f seconds.\n" % dur)

        self.wrench = extract_raw_forces(self.data)

        self.force = extract_tcp_forces(self.data)
        self.force_xyz, self.force_rvec = np.split(self.force, 2, axis=1)
        self.force_rmats = rotations.rvecs2matrices(self.force_rvec)
        self.force_euler = rotations.matrix2euler(self.force_rmats)

        self.raw_force = extract_raw_forces(self.data)
        self.raw_force_xyz, self.raw_force_rvec = np.split(self.raw_force, 2, axis=1)
        self.raw_force_rmats = rotations.rvecs2matrices(self.raw_force_rvec)
        self.raw_force_euler = rotations.matrix2euler(self.raw_force_rmats)

        self.pose = extract_tcp_poses(self.data)
        self.pose_xyz, self.pose_rvec = np.split(self.pose, 2, axis=1)
        self.pose_rmats = rotations.rvecs2matrices(self.pose_rvec)
        self.pose_euler = rotations.matrix2euler(self.pose_rmats)

        self.speed = extract_tcp_speed(self.data)
        self.speed_xyz, self.speed_rvec = np.split(self.speed, 2, axis=1)
        self.speed_rmats = rotations.rvecs2matrices(self.speed_rvec)
        self.speed_euler = rotations.matrix2euler(self.speed_rmats)

        self.joint_radians = extract_joint_angles(self.data)
    
    def purge_dynamic(self, epsilon=0.001, verbose=True):
        """ Remove all time steps in which the robot was moving. """

        speed_xyz = np.linalg.norm(self.speed_xyz, axis=1)
        speed_rvec = np.linalg.norm(self.speed_rvec, axis=1)

        still = speed_xyz + speed_rvec < epsilon

        self.force = self.force[still, :]
        self.force_xyz = self.force_xyz[still, :]
        self.force_rvec = self.force_rvec[still, :]
        self.force_rmats = self.force_rmats[still, :]
        self.force_euler = self.force_euler[still, :]

        self.pose = self.pose[still, :]
        self.pose_xyz = self.pose_xyz[still, :]
        self.pose_rvec = self.pose_rvec[still, :]
        self.pose_rmats = self.pose_rmats[still, :]
        self.pose_euler = self.pose_euler[still, :]

        self.speed = self.speed[still, :]
        self.speed_xyz = self.speed_xyz[still, :]
        self.speed_rvec = self.speed_rvec[still, :]
        self.speed_rmats = self.speed_rmats[still, :]
        self.speed_euler = self.speed_euler[still, :]

        self.joint_radians = self.joint_radians[still, :]

        print("Removed %s / %s frames.\n" % (sum(~still), len(still)))

        return self
