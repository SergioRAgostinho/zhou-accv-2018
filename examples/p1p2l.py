import numpy as np
from zhou_accv_2018 import p1p2l

# fix seed to allow for reproducible results
np.random.seed(0)
np.random.seed(42)

# instantiate a couple of points centered around the origin
pts = 0.6 * (np.random.random((1, 3)) - 0.5)

# 3D lines are parameterized as pts and direction stacked into a tuple
# instantiate a couple of points centered around the origin
pts_l = 0.6 * (np.random.random((2, 3)) - 0.5)
# generate normalized directions
directions = 2 * (np.random.random((2, 3)) - 0.5)
directions /= np.linalg.norm(directions, axis=1)[:, None]

line_3d = (pts_l, directions)

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

# sample 2 points from each line and stack all
pts_ls = np.hstack((pts_l, pts_l + directions)).reshape((-1, 3))
pts_all = np.vstack((pts, pts_ls))

# Project everything to 2D
pts_all_2d = (pts_all @ R_gt.T + t_gt) @ K.T
pts_all_2d = (pts_all_2d / pts_all_2d[:, -1, None])[:, :-1]

pts_2d = pts_all_2d[:1]
line_2d = pts_all_2d[1:].reshape((-1, 2, 2))

# Compute pose candidates. the problem is not minimal so only one
# will be provided
poses = p1p2l(pts_2d=pts_2d, line_2d=line_2d, pts_3d=pts, line_3d=line_3d, K=K)

# The error criteria for lines is to ensure that both 3D points and
# direction, after transformation, are inside the plane formed by the
# line projection. We start by computing the plane normals

# line in 2D has two sampled points.
line_2d_c = np.linalg.solve(
    K, np.vstack((line_2d.reshape((2 * 2, 2)).T, np.ones((1, 2 * 2))))
).T
line_2d_c = line_2d_c.reshape((2, 2, 3))

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

    # Project points to 2D
    pts_2d_est = (pts @ R.T + t) @ K.T
    pts_2d_est = (pts_2d_est / pts_2d_est[:, -1, None])[:, :-1]
    err_p = np.mean(np.linalg.norm(pts_2d - pts_2d_est, axis=1))

    # pts_l
    pts_l_est = pts_l @ R.T + t
    err_l_pt = np.mean(np.abs(np.sum(pts_l_est * n_li, axis=1)))

    # directions
    dir_est = directions @ R.T
    err_l_dir = np.mean(
        np.arcsin(np.abs(np.sum(dir_est * n_li, axis=1))) * 180.0 / np.pi
    )

    print("Estimate -", i + 1)
    print("R (estimate):", R, sep="\n")
    print("t (estimate):", t)
    print("Mean error (pixels):", err_p)
    print("Mean pt distance from plane (m):", err_l_pt)
    print("Mean angle error from plane (°):", err_l_dir)
