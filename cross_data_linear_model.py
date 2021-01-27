import os
import json
import numpy as np
from matplotlib import pyplot as plt

import read
import approximation


def meanouter(x, y):
    outer = x[:, :, None] * y[:, None, :]
    return np.mean(outer, axis=0)


def cov(x, y):
    xbar = x - np.mean(x, axis=0)
    ybar = y - np.mean(y, axis=0)
    return meanouter(xbar, ybar)


def get_waves(angles, frequencies):
    assert np.ndim(angles) == 2
    assert np.ndim(frequencies) == 1
    cos = [np.cos(f * angles) for f in frequencies]
    sin = [np.sin(f * angles) for f in frequencies]
    return np.concatenate(cos + sin, axis=1)


jsonpath = "/Users/mathias/Downloads/data.json"
with open(jsonpath, "r") as source:
    data1 = json.load(source)

jsonpath = "/Users/mathias/Downloads/data20jan2021.json"
with open(jsonpath) as source:
    blocks = source.read().split("---")
    data2 = [json.loads(b.strip()) for b in blocks]


def build_features(subset, degree):
    joint5angles = np.array(subset['angles'])
    jointtemps = np.array(subset['temps'])  # temperatures of ALL joints
    joint5temps = jointtemps[:, 5]
    w = get_waves(joint5angles[:, None], np.arange(NUMFREQS))
    t = joint5temps[:, None]
    return np.concatenate([w * t**d for d in range(degree + 1)], axis=1)


NUMFREQS = 10
DEGREE = 1

x1 = np.concatenate([build_features(subset, DEGREE)
                     for subset in data1], axis=0)

x2 = np.concatenate([build_features(subset, DEGREE)
                     for subset in data2], axis=0)

y1 = np.concatenate([np.array(subset['forces'])[:, :3]
                     for subset in data1], axis=0)

y2 = np.concatenate([np.array(subset['forces'])[:, :3]
                     for subset in data2], axis=0)

slope1 = meanouter(y1, x1) @ np.linalg.pinv(meanouter(x1, x1))
slope2 = meanouter(y2, x2) @ np.linalg.pinv(meanouter(x2, x2))

yhat11 = x1 @ slope1.T
yhat12 = x1 @ slope2.T
yhat21 = x2 @ slope1.T
yhat22 = x2 @ slope2.T

errors11 = np.mean((y1 - yhat11) ** 2, axis=0)
errors12 = np.mean((y1 - yhat12) ** 2, axis=0)
errors21 = np.mean((y2 - yhat21) ** 2, axis=0)
errors22 = np.mean((y2 - yhat22) ** 2, axis=0)