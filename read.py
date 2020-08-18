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


def extract_tcp_speed(data):
    """ Extract the shape-(N, 6) array of TCP speeds. """

    names = ["actual_TCP_speed_%s" % i for i in range(6)]
    columns = [data[name] for name in names]

    return np.stack(columns, axis=1)


class Recording:

    def __init__(self, csvpath):

        with open(csvpath, "r") as source:
            self.data = text2dict(source.read())

        self.force = extract_tcp_forces(self.data)
        self.force_xyz, self.force_rvec = np.split(self.force, 2, axis=1)
        self.force_rmats = rotations.rvecs2matrices(self.force_rvec)
        self.force_euler = rotations.matrix2euler(self.force_rmats)

        self.pose = extract_tcp_poses(self.data)
        self.pose_xyz, self.pose_rvec = np.split(self.pose, 2, axis=1)
        self.pose_rmats = rotations.rvecs2matrices(self.pose_rvec)
        self.pose_euler = rotations.matrix2euler(self.pose_rmats)

        self.speed = extract_tcp_speed(self.data)
        self.speed_xyz, self.speed_rvec = np.split(self.speed, 2, axis=1)
        self.speed_rmats = rotations.rvecs2matrices(self.speed_rvec)
        self.speed_euler = rotations.matrix2euler(self.speed_rmats)
