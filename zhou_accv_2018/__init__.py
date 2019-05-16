import numpy as np

from .e3q3 import e3q3

"""An implementation of the registration problem from "A Stable Algebraic Camera
Pose Estimation for Minimal Configurations of 2D/3D Point and Line Correspondences",
from Zhou et al. at ACCV 2018.
"""

__version__ = "1.0.0"


def _re3q3(A, q_ref=None, allow_imag_roots=False):
    """Robust E3Q3 is a robust implementation of "Efficient Intersection of Three Quadrics
    and Applications in Computer Viion" (E3Q3) by Kukelova et al. as suggested in the paper
    "Minimal Solution of 2D/3D Point and Line Correspondences" by Zhou et al.

    A - the 3x10 coefficient matrix
    q_ref - a reference quaternion [w, x, y, z], to perform the coefficient swap for
    increased stability. Default: None
    allow_imag_roots - allows imaginary roots (solutions) by only considering
    their real part. Recommended to set to True under noise conditions.
    Default: False
    """

    ## Permutations to solve the inversion problem of the w = 0 quaternion coefficient

    if q_ref is None:
        q_ref = np.array([1, 0, 0, 0])

    q_swap_idx = np.argmax(np.abs(q_ref))

    # Original
    # [x^2, y^2, z^2, w^2, xy, xz, xw, yz, yw, zw]
    # with x = aw , y = bw, z = cw and dividing by w^2
    # to reorder to

    # Permute coefficients in x
    # [w^2, y^2, z^2, x^2, wy, wz, wx, yz, yx, zx]
    # requires the following permutation
    # [3, 1, 2, 0, 8, 9, 6, 7, 4, 5]

    # Permute coefficients in y
    # [x^2, w^2, z^2, y^2, xw, xz, xy, wz, wy, zy]
    # requires the following permutation
    # [0, 3, 2, 1, 6, 5, 4, 9, 8, 7]

    # Permute coefficients in z
    # [x^2, y^2, w^2, z^2, xy, xw, xz, yw, yz, wz]
    # requires the following permutation
    # [0, 1, 3, 2, 4, 6, 5, 8, 7, 9]

    perms = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [3, 1, 2, 0, 8, 9, 6, 7, 4, 5],
        [0, 3, 2, 1, 6, 5, 4, 9, 8, 7],
        [0, 1, 3, 2, 4, 6, 5, 8, 7, 9],
    ]

    A_r = (A.T)[perms[q_swap_idx]].T

    ## Permutations to solve the inversion problem of Matrix H

    # Original
    # [x^2, y^2, z^2, w^2, xy, xz, xw, yz, yw, zw]
    # with x = aw , y = bw, z = cw and dividing by w^2
    # to reorder to

    # Permute coefficients in A
    # [a^2, b^2, c^2, ab, ac, bc, a, b, c, 1]
    # requires the following permutation
    # [0, 1, 2, 4, 5, 7, 6, 8, 9, 3]
    A_r = (A_r.T)[[0, 1, 2, 4, 5, 7, 6, 8, 9, 3]].T

    # Invoke the E3Q3
    a, b, c = e3q3(A_r, allow_imag_roots)

    # Retrieve quaternion's coefficients
    w = np.sqrt(1 / (a * a + b * b + c * c + 1))  # there was mistake here :D it's w2
    x = a * w
    y = b * w
    z = c * w

    q_prime = np.array([w, x, y, z])
    perms = [[0, 1, 2, 3], [1, 0, 2, 3], [2, 1, 0, 3], [3, 1, 2, 0]]

    w, x, y, z = q_prime[perms[q_swap_idx]]

    # Compose the final rotation matrix
    R11 = w * w + x * x - y * y - z * z
    R12 = 2 * (x * y - w * z)
    R13 = 2 * (w * y + x * z)

    R21 = 2 * (x * y + w * z)
    R22 = w * w - x * x + y * y - z * z
    R23 = 2 * (y * z - w * x)

    R31 = 2 * (x * z - w * y)
    R32 = 2 * (y * z + w * x)
    R33 = w * w - x * x - y * y + z * z

    R = np.block(
        [
            [R11[:, None, None], R12[:, None, None], R13[:, None, None]],
            [R21[:, None, None], R22[:, None, None], R23[:, None, None]],
            [R31[:, None, None], R32[:, None, None], R33[:, None, None]],
        ]
    )

    # Compose the final r vector
    r = np.stack(
        (x * x, y * y, z * z, w * w, x * y, x * z, x * w, y * z, y * w, z * w), axis=1
    )

    return R, r


def _re3q3_point_constraints(pts_2d, pts_3d, K):

    n = len(pts_3d)

    # Expand arguments
    # points in 2D
    px, py, pz = np.linalg.solve(K, np.vstack((pts_2d.T, np.ones(n))))

    # points in 3D
    Px, Py, Pz = pts_3d.T

    # Point Constraints
    # Matrix([
    #     [-2*Px*py*w*y + 2*Px*py*x*z - 2*Px*pz*w*z - 2*Px*pz*x*y + 2*Py*py*w*x + 2*Py*py*y*z - Py*pz*w**2 + Py*pz*x**2 - Py*pz*y**2 + Py*pz*z**2 + Pz*py*w**2 - Pz*py*x**2 - Pz*py*y**2 + Pz*py*z**2 + 2*Pz*pz*w*x - 2*Pz*pz*y*z + py*tz - pz*ty],
    #     [ 2*Px*px*w*y - 2*Px*px*x*z + Px*pz*w**2 + Px*pz*x**2 - Px*pz*y**2 - Px*pz*z**2 - 2*Py*px*w*x - 2*Py*px*y*z - 2*Py*pz*w*z + 2*Py*pz*x*y - Pz*px*w**2 + Pz*px*x**2 + Pz*px*y**2 - Pz*px*z**2 + 2*Pz*pz*w*y + 2*Pz*pz*x*z - px*tz + pz*tx],
    #     [ 2*Px*px*w*z + 2*Px*px*x*y - Px*py*w**2 - Px*py*x**2 + Px*py*y**2 + Px*py*z**2 + Py*px*w**2 - Py*px*x**2 + Py*px*y**2 - Py*px*z**2 + 2*Py*py*w*z - 2*Py*py*x*y - 2*Pz*px*w*x + 2*Pz*px*y*z - 2*Pz*py*w*y - 2*Pz*py*x*z + px*ty - py*tx],
    # ])

    Pxpx = Px * px
    Pxpy = Px * py
    Pxpz = Px * pz
    Pypx = Py * px
    Pypy = Py * py
    Pypz = Py * pz
    Pzpx = Pz * px
    Pzpy = Pz * py
    Pzpz = Pz * pz

    c11 = Pypz - Pzpy  # x**2
    c12 = -Pypz - Pzpy  # y**2
    c13 = Pypz + Pzpy  # z**2
    c14 = -Pypz + Pzpy  # w**2
    c15 = -2 * Pxpz  # x*y
    c16 = 2 * Pxpy  # x*z
    c17 = 2 * Pypy + 2 * Pzpz  # w*x
    c18 = 2 * Pypy - 2 * Pzpz  # y*z
    c19 = -2 * Pxpy  # w*y
    c110 = -2 * Pxpz  # w*z

    c21 = Pxpz + Pzpx  # x**2
    c22 = -Pxpz + Pzpx  # y**2
    c23 = -Pxpz - Pzpx  # z**2
    c24 = Pxpz - Pzpx  # w**2
    c25 = 2 * Pypz  # x*y
    c26 = -2 * Pxpx + 2 * Pzpz  # x*z
    c27 = -2 * Pypx  # w*x
    c28 = -2 * Pypx  # y*z
    c29 = 2 * Pxpx + 2 * Pzpz  # w*y
    c210 = -2 * Pypz  # w*z

    n11 = np.zeros(n)
    n12 = -pz
    n13 = py

    n21 = pz
    n22 = np.zeros(n)
    n23 = -px

    ## Compose block matrices for the equation system
    c1 = np.stack((c11, c12, c13, c14, c15, c16, c17, c18, c19, c110), axis=1)
    c2 = np.stack((c21, c22, c23, c24, c25, c26, c27, c28, c29, c210), axis=1)
    n1 = np.stack((n11, n12, n13), axis=1)
    n2 = np.stack((n21, n22, n23), axis=1)
    c3 = None
    n3 = None

    # Account for the special 3 points case
    if n == 3:

        c31 = -Pxpy - Pypx  # x**2
        c32 = Pxpy + Pypx  # y**2
        c33 = Pxpy - Pypx  # z**2
        c34 = -Pxpy + Pypx  # w**2
        c35 = 2 * Pxpx - 2 * Pypy  # x*y
        c36 = -2 * Pzpy  # x*z
        c37 = -2 * Pzpx  # w*x
        c38 = 2 * Pzpx  # y*z
        c39 = -2 * Pzpy  # w*y
        c310 = 2 * Pxpx + 2 * Pypy  # w*z

        n31 = -py
        n32 = px
        n33 = np.zeros(n)

        c3 = np.stack((c31, c32, c33, c34, c35, c36, c37, c38, c39, c310), axis=1)
        n3 = np.stack((n31, n32, n33), axis=1)

    return (c1, c2, c3), (n1, n2, n3)


def _re3q3_line_constraints(line_2d, line_3d, K):

    n = len(line_2d)

    # line in 2D
    # has two sampled points.
    line_2d_c = np.linalg.solve(
        K, np.vstack((line_2d.reshape((2 * n, 2)).T, np.ones((1, 2 * n))))
    ).T
    line_2d_c = line_2d_c.reshape((n, 2, 3))

    # row wise cross product
    n_li = np.cross(line_2d_c[:, 0, :], line_2d_c[:, 1, :])

    # Normalize for stability
    n_li /= np.linalg.norm(n_li, axis=1)[:, None]

    nx, ny, nz = n_li.T

    # line in 3D
    PL, vL = line_3d
    PLx, PLy, PLz = PL.T
    vLx, vLy, vLz = vL.T

    # Line Constraints - direction
    # Matrix([[nx*vLx*w**2 + nx*vLx*x**2 - nx*vLx*y**2 - nx*vLx*z**2 - 2*nx*vLy*w*z + 2*nx*vLy*x*y + 2*nx*vLz*w*y + 2*nx*vLz*x*z + 2*ny*vLx*w*z + 2*ny*vLx*x*y + ny*vLy*w**2 - ny*vLy*x**2 + ny*vLy*y**2 - ny*vLy*z**2 - 2*ny*vLz*w*x + 2*ny*vLz*y*z - 2*nz*vLx*w*y + 2*nz*vLx*x*z + 2*nz*vLy*w*x + 2*nz*vLy*y*z + nz*vLz*w**2 - nz*vLz*x**2 - nz*vLz*y**2 + nz*vLz*z**2]])

    nxvLx = nx * vLx
    nxvLy = nx * vLy
    nxvLz = nx * vLz
    nyvLx = ny * vLx
    nyvLy = ny * vLy
    nyvLz = ny * vLz
    nzvLx = nz * vLx
    nzvLy = nz * vLy
    nzvLz = nz * vLz

    cl11 = nxvLx - nyvLy - nzvLz  # x**2
    cl12 = -nxvLx + nyvLy - nzvLz  # y**2
    cl13 = -nxvLx - nyvLy + nzvLz  # z**2
    cl14 = nxvLx + nyvLy + nzvLz  # w**2
    cl15 = 2 * nxvLy + 2 * nyvLx  # x*y
    cl16 = 2 * nxvLz + 2 * nzvLx  # x*z
    cl17 = -2 * nyvLz + 2 * nzvLy  # w*x
    cl18 = 2 * nyvLz + 2 * nzvLy  # y*z
    cl19 = 2 * nxvLz - 2 * nzvLx  # w*y
    cl110 = -2 * nxvLy + 2 * nyvLx  # w*z

    # Line Constraints - point
    # n.T*(R*PL + t)
    # Matrix([[PLx*nx*w**2 + PLx*nx*x**2 - PLx*nx*y**2 - PLx*nx*z**2 + 2*PLx*ny*w*z + 2*PLx*ny*x*y - 2*PLx*nz*w*y + 2*PLx*nz*x*z - 2*PLy*nx*w*z + 2*PLy*nx*x*y + PLy*ny*w**2 - PLy*ny*x**2 + PLy*ny*y**2 - PLy*ny*z**2 + 2*PLy*nz*w*x + 2*PLy*nz*y*z + 2*PLz*nx*w*y + 2*PLz*nx*x*z - 2*PLz*ny*w*x + 2*PLz*ny*y*z + PLz*nz*w**2 - PLz*nz*x**2 - PLz*nz*y**2 + PLz*nz*z**2 + nx*tx + ny*ty + nz*tz]])

    PLxnx = PLx * nx
    PLxny = PLx * ny
    PLxnz = PLx * nz
    PLynx = PLy * nx
    PLyny = PLy * ny
    PLynz = PLy * nz
    PLznx = PLz * nx
    PLzny = PLz * ny
    PLznz = PLz * nz

    cl21 = PLxnx - PLyny - PLznz  # x**2
    cl22 = -PLxnx + PLyny - PLznz  # y**2
    cl23 = -PLxnx - PLyny + PLznz  # z**2
    cl24 = PLxnx + PLyny + PLznz  # w**2
    cl25 = 2 * PLxny + 2 * PLynx  # x*y
    cl26 = 2 * PLxnz + 2 * PLznx  # x*z
    cl27 = 2 * PLynz - 2 * PLzny  # w*x
    cl28 = 2 * PLynz + 2 * PLzny  # y*z
    cl29 = -2 * PLxnz + 2 * PLznx  # w*y
    cl210 = 2 * PLxny - 2 * PLynx  # w*z

    ## Compose block matrices for the equation system
    cl1 = np.stack(
        (cl11, cl12, cl13, cl14, cl15, cl16, cl17, cl18, cl19, cl110), axis=1
    )
    cl2 = np.stack(
        (cl21, cl22, cl23, cl24, cl25, cl26, cl27, cl28, cl29, cl210), axis=1
    )

    return (cl1, cl2), n_li


def p3p(pts_2d, pts_3d, K, q_ref=None, allow_imag_roots=False):
    """An implementation of the ... problem from "A Stable Algebraic Camera
    Pose Estimation for Minimal Configurations of 2D/3D Point and Line Correspondences",
    from Zhou et al. at ACCV 2018.

    3 points

    pts_2d - pixels in 2d. Each pixel is a row.
    pts_3d - the corresponding points in 3D
    K - the intrinsics matrix of the camera
    q_ref - [w, x, y, z] an initial coarse estimate of the rotation to improve robustness
    allow_imag_roots - allows imaginary roots (solutions) by only considering
    their real part. Recommended to set to True under noise conditions.
    Default: False
    """

    # Extract point constraints
    (C1, c2, c3), (N1, n2, n3) = _re3q3_point_constraints(pts_2d, pts_3d, K)

    # Compose block matrices
    C2 = np.vstack((c2[:2], c3[2]))
    N2 = np.vstack((n2[:2], n3[2]))

    N2_inv = np.linalg.inv(N2)
    A = C1 - N1 @ N2_inv @ C2

    # Invoke the modified re3q3
    R, r = _re3q3(A=A, q_ref=q_ref, allow_imag_roots=allow_imag_roots)

    # Getting the translation components back
    t = -(N2_inv @ C2 @ r.T).T

    # send everything back
    return list(zip(R, t))


def p2p1l(pts_2d, line_2d, pts_3d, line_3d, K, q_ref=None, allow_imag_roots=False):
    """An implementation of the ... problem from "A Stable Algebraic Camera
    Pose Estimation for Minimal Configurations of 2D/3D Point and Line Correspondences",
    from Zhou et al. at ACCV 2018.

    2 points 1 line variant

    pts_2d - pixels in 2d. Each pixel is a row.
    line_2d - two pixels sampled  from the line. each pixels is a row
    pts_3d - the corresponding points in 3D
    line_3d - a 3D line defined by a tuple whose first elements defines a point
    in 3d space and second a direction in 3D.
    K - the intrinsics matrix of the camera
    q_ref - [w, x, y, z] an initial coarse estimate of the rotation to improve robustness
    allow_imag_roots - allows imaginary roots (solutions) by only considering
    their real part. Recommended to set to True under noise conditions.
    Default: False

    All direction are assumed to be normalized.
    """

    # Extract point constraints
    (C1, c2, _), (N1, n2, _) = _re3q3_point_constraints(pts_2d, pts_3d, K)

    # Extract line constraints
    (cl1, cl2), n_li = _re3q3_line_constraints(line_2d.reshape((1, 2, 2)), line_3d, K)

    ## Compose block matrices for the equation system
    C2 = np.vstack((cl2, c2))
    N2 = np.vstack((n_li, n2))
    N2_inv = np.linalg.inv(N2)

    A = np.vstack((cl1, C1 - N1 @ N2_inv @ C2))

    # Invoke the modified re3q3
    R, r = _re3q3(A=A, q_ref=q_ref, allow_imag_roots=allow_imag_roots)

    # Getting the translation components back
    t = -(N2_inv @ C2 @ r.T).T

    # send everything back
    return list(zip(R, t))


def p1p2l(pts_2d, line_2d, pts_3d, line_3d, K, q_ref=None, allow_imag_roots=False):
    """An implementation of the ... problem from "A Stable Algebraic Camera
    Pose Estimation for Minimal Configurations of 2D/3D Point and Line Correspondences",
    from Zhou et al. at ACCV 2018.

    1 point 2 lines variant

    pts_2d - pixel in 2d.
    line_2d - A 2x2x2 tensor with two pixels sampled from each line. line x point x pixel
    pts_3d - the corresponding points in 3D
    line_3d - a 3D line defined by a tuple whose first elements defines points
    in 3d space and second the directions in 3D.
    K - the intrinsics matrix of the camera
    q_ref - [w, x, y, z] an initial coarse estimate of the rotation to improve robustness
    allow_imag_roots - allows imaginary roots (solutions) by only considering
    their real part. Recommended to set to True under noise conditions.
    Default: False

    All direction are assumed to be normalized.
    """

    # Extract point constraints
    (c1, c2, _), (n1, n2, _) = _re3q3_point_constraints(
        pts_2d=pts_2d.reshape((1, 2)), pts_3d=pts_3d.reshape((1, 3)), K=K
    )

    # Extract line constraints
    (cl1, cl2), n_li = _re3q3_line_constraints(line_2d, line_3d, K)

    ## Compose block matrices for the equation system
    C2 = np.vstack((cl2, c2))
    N2 = np.vstack((n_li, n2))
    N2_inv = np.linalg.inv(N2)

    A = np.vstack((cl1, c1 - n1 @ N2_inv @ C2))

    # Invoke the modified re3q3
    R, r = _re3q3(A=A, q_ref=q_ref, allow_imag_roots=allow_imag_roots)

    # Getting the translation components back
    t = -(N2_inv @ C2 @ r.T).T

    # send everything back
    return list(zip(R, t))


def p3l(line_2d, line_3d, K, q_ref=None, allow_imag_roots=False):
    """An implementation of the ... problem from "A Stable Algebraic Camera
    Pose Estimation for Minimal Configurations of 2D/3D Point and Line Correspondences",
    from Zhou et al. at ACCV 2018.

    3 lines variant

    line_2d - A 3x2x2 tensor with two pixels sampled from each line. line x point x pixel
    line_3d - a 3D line defined by a tuple whose first element defines the points
    in 3d space and the second a directions in 3D.
    K - the intrinsics matrix of the camera
    q_ref - [w, x, y, z] an initial coarse estimate of the rotation to improve robustness
    allow_imag_roots - allows imaginary roots (solutions) by only considering
    their real part. Recommended to set to True under noise conditions.
    Default: False
    """

    ## Compose block matrices for the equation system
    # A = np.stack((cl11, cl12, cl13, cl14, cl15, cl16, cl17, cl18, cl19, cl110), axis=1)
    # C2 = np.stack((cl21, cl22, cl23, cl24, cl25, cl26, cl27, cl28, cl29, cl210), axis=1)
    # N2 = n_li
    (A, C2), N2 = _re3q3_line_constraints(line_2d, line_3d, K)

    # Invoke the modified re3q3
    R, r = _re3q3(A=A, q_ref=q_ref, allow_imag_roots=allow_imag_roots)

    # Getting the translation components back
    t = -np.linalg.solve(N2, C2 @ r.T).T

    # send everything back
    return list(zip(R, t))
