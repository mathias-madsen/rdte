import numpy as np


def text2dataset(text):
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
