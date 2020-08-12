import warnings
import numpy as np
from pyquaternion import Quaternion


def rvecs2quats(rvecs):
    """ Convert array of rotation vectors to list of Quaternions. """

    norms = np.linalg.norm(rvecs, axis=1)
    axes = rvecs + 1e-14*np.array([1., 0., 0.])

    return [Quaternion(axis=axis, radians=norm)
             for axis, norm in zip(axes, norms)]


def rvecs2matrices(rvecs):
    """ Convert an array of rotation vectors to rotation matrices. """

    quats = rvecs2quats(rvecs)

    return np.stack([q.rotation_matrix for q in quats])


def quats2rvecs(quaternions):
    """ Convert list of quaternions to array of rotation vectors. """

    return np.stack([q.radians * q.axis for q in quaternions])


def relativize_rvecs(rvecs):
    """ Make an array of rotations relative to the first. """

    quats = rvecs2quats(rvecs)
    first_quat_inverse = quats[0].inverse
    relquats = [first_quat_inverse * q for q in quats]

    return np.array([q.radians * q.axis for q in relquats])


def matrix2euler(matrix):
    """ Convert a rotation matrix to intrinsic Euler angles.

    Based on https://github.com/scipy/scipy/blob/v1.5.2/scipy/spatial/
    transform/rotation.py#L18-L140, which in turn in based on "General
    Formula for Extractingthe Euler Angles" (Shuster and Markley, 2006).
    """

    if matrix.ndim == 2:
        matrix = matrix[None, :, :]

    num_rotations = matrix.shape[0]

    # step 1
    n3, n2, n1 = np.eye(3)

    # Step 2
    sl = np.dot(np.cross(n1, n2), n3)
    cl = np.dot(n1, n3)
    offset = np.arctan2(sl, cl)
    c = np.vstack((n2, np.cross(n1, n2), n1))

    # Step 3
    rot = np.array([
        [1, 0, 0],
        [0, cl, sl],
        [0, -sl, cl],
    ])
    res = np.einsum('...ij,...jk->...ik', c, matrix)
    transformed = np.einsum('...ij,...jk->...ik', res, c.T.dot(rot))

    # Step 4
    angles = np.empty((num_rotations, 3))
    # Ensure less than unit norm
    positive_unity = transformed[:, 2, 2] > 1
    negative_unity = transformed[:, 2, 2] < -1
    transformed[positive_unity, 2, 2] = 1
    transformed[negative_unity, 2, 2] = -1
    angles[:, 1] = np.arccos(transformed[:, 2, 2])

    # Steps 5, 6
    eps = 1e-7
    safe1 = (np.abs(angles[:, 1]) >= eps)
    safe2 = (np.abs(angles[:, 1] - np.pi) >= eps)

    # Step 4 (Completion)
    angles[:, 1] += offset

    # 5b
    safe_mask = np.logical_and(safe1, safe2)
    angles[safe_mask, 0] = np.arctan2(transformed[safe_mask, 0, 2],
                                      -transformed[safe_mask, 1, 2])
    angles[safe_mask, 2] = np.arctan2(transformed[safe_mask, 2, 0],
                                      transformed[safe_mask, 2, 1])

    # Set first angle to zero so that after reversal we
    # ensure that third angle is zero
    # 6a
    angles[~safe_mask, 0] = 0
    # 6b
    angles[~safe1, 2] = np.arctan2(transformed[~safe1, 1, 0]
                                   - transformed[~safe1, 0, 1],
                                   transformed[~safe1, 0, 0]
                                   + transformed[~safe1, 1, 1])
    # 6c
    angles[~safe2, 2] = -np.arctan2(transformed[~safe2, 1, 0]
                                    + transformed[~safe2, 0, 1],
                                    transformed[~safe2, 0, 0])

    # Step 7
    if np.allclose(n1, n3):
        # lambda = 0, so we can only ensure angle2 -> [0, pi]
        adjust_mask = np.logical_or(angles[:, 1] < 0, angles[:, 1] > np.pi)
    else:
        # lambda = + or - pi/2, so we can ensure angle2 -> [-pi/2, pi/2]
        adjust_mask = np.logical_or(angles[:, 1] < -np.pi / 2,
                                    angles[:, 1] > np.pi / 2)

    # Dont adjust gimbal locked angle sequences
    adjust_mask = np.logical_and(adjust_mask, safe_mask)

    angles[adjust_mask, 0] += np.pi
    angles[adjust_mask, 1] = 2 * offset - angles[adjust_mask, 1]
    angles[adjust_mask, 2] -= np.pi

    angles[angles < -np.pi] += 2 * np.pi
    angles[angles > np.pi] -= 2 * np.pi

    # Step 8
    if not np.all(safe_mask):
        warnings.warn("Gimbal lock detected. Setting third angle to"
                      " zero since it is not possible to uniquely "
                      "determine all angles.")

    # Reverse role of extrinsic and intrinsic rotations,
    # but let third angle be zero for gimbal locked cases
    angles = angles[:, ::-1]

    return angles
