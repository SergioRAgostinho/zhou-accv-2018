import numpy as np
from zhou_accv_2018 import p3p

# fix seed to allow for reproducible results
np.random.seed(0)
np.random.seed(42)

# instantiate a couple of points centered around the origin
pts = 0.6 * (np.random.random((3, 3)) - 0.5)

# Made up projective matrix
K = np.array([[160, 0, 320], [0, 120, 240], [0, 0, 1]])

# A pose
R_gt = np.array(
    [
        [-0.48048015, 0.1391384, -0.86589799],
        [-0.0333282, -0.98951829, -0.14050899],
        [-0.8763721, -0.03865296, 0.48008113],
    ]
)
t_gt = np.array([-0.10266772, 0.25450789, 1.70391109])

# Project points to 2D
pts_2d = (pts @ R_gt.T + t_gt) @ K.T
pts_2d = (pts_2d / pts_2d[:, -1, None])[:, :-1]

# Compute pose candidates. the problem is not minimal so only one
# will be provided
poses = p3p(pts_2d=pts_2d, pts_3d=pts, K=K)

# Print results
print("R (ground truth):", R_gt, sep="\n")
print("t (ground truth):", t_gt)

print("Nr of possible poses:", len(poses))
for i, pose in enumerate(poses):
    R, t = pose

    # Project points to 2D
    pts_2d_est = (pts @ R.T + t) @ K.T
    pts_2d_est = (pts_2d_est / pts_2d_est[:, -1, None])[:, :-1]
    err = np.mean(np.linalg.norm(pts_2d - pts_2d_est, axis=1))

    print("Estimate -", i + 1)
    print("R (estimate):", R, sep="\n")
    print("t (estimate):", t)
    print("Mean error (pixels):", err)
