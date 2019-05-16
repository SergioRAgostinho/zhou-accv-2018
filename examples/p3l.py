import numpy as np
from zhou_accv_2018 import p3l

# fix seed to allow for reproducible results
np.random.seed(0)
np.random.seed(42)

# 3D lines are parameterized as pts and direction stacked into a tuple
# instantiate a couple of points centered around the origin
pts = 0.6 * (np.random.random((3, 3)) - 0.5)

# generate normalized directions
directions = 2 * (np.random.random((3, 3)) - 0.5)
directions /= np.linalg.norm(directions, axis=1)[:, None]

line_3d = (pts, directions)

# Made up projective matrix
K = np.array([[160, 0, 320], [0, 120, 240], [0, 0, 1]])

# A pose
R_gt = np.array(
    [
        [0.89802142, -0.41500101, 0.14605372],
        [0.24509948, 0.7476071, 0.61725997],
        [-0.36535431, -0.51851499, 0.77308372],
    ]
)
t_gt = np.array([-0.0767557, 0.13917375, 1.9708239])

# Sample to points from the line and project them to 2D
pts_s = np.hstack((pts, pts + directions)).reshape((-1, 3))
line_2d = (pts_s @ R_gt.T + t_gt) @ K.T

# this variable is organized as (line, point, dim)
line_2d = (line_2d / line_2d[:, -1, None])[:, :-1].reshape((-1, 2, 2))

# Compute pose candidates. the problem is not minimal so only one
# will be provided
poses = p3l(line_2d=line_2d, line_3d=line_3d, K=K)

# The error criteria for lines is to ensure that both 3D points and
# direction, after transformation, are inside the plane formed by the
# line projection. We start by computing the plane normals

# line in 2D has two sampled points.
line_2d_c = np.linalg.solve(
    K, np.vstack((line_2d.reshape((2 * 3, 2)).T, np.ones((1, 2 * 3))))
).T
line_2d_c = line_2d_c.reshape((3, 2, 3))

# row wise cross product + normalization
n_li = np.cross(line_2d_c[:, 0, :], line_2d_c[:, 1, :])
n_li /= np.linalg.norm(n_li, axis=1)[:, None]

# Print results
print("R (ground truth):", R_gt, sep="\n")
print("t (ground truth):", t_gt)
print("Nr of possible poses:", len(poses))
for i, pose in enumerate(poses):
    R, t = pose

    # The error criteria for lines is to ensure that both 3D points and
    # direction, after transformation, are inside the plane formed by the
    # line projection

    # pts
    pts_est = pts @ R.T + t
    err_pt = np.mean(np.abs(np.sum(pts_est * n_li, axis=1)))

    # directions
    dir_est = directions @ R.T
    err_dir = np.mean(np.arcsin(np.abs(np.sum(dir_est * n_li, axis=1))) * 180.0 / np.pi)

    print("Estimate -", i + 1)
    print("R (estimate):", R, sep="\n")
    print("t (estimate):", t)
    print("Mean pt distance from plane (m):", err_pt)
    print("Mean angle error from plane (°):", err_dir)
