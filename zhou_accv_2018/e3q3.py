import numpy as np

from .e3q3c import coefficients


def e3q3(A, allow_imag_roots=False):
    """Robust E3Q3 is a robust implementation of "Efficient Intersection of Three Quadrics
    and Applications in Computer Vision" (E3Q3) by Kukelova et al. as suggested in the paper
    "Minimal Solution of 2D/3D Point and Line Correspondences" by Zhou et al.

    A - the 3x10 coefficient matrix, with A @ z = 0 and assuming a quadratic structure
    of z = [a**2, b**2, c**2, ab, ac, bc, a, b, c, 1]
    allow_imag_roots - allows imaginary roots (solutions) by only considering
    their real part. Recommended to set to True under noise conditions.
    Default: False
    """

    ## Permutations to solve the inversion problem of Matrix H

    # Permute coefficients in A
    # [a^2, b^2, c^2, ab, ac, bc, a, b, c, 1]
    # requires the following permutation
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Permute coefficients in B
    # [b^2, c^2, a^2, bc, ba, ca, b, c, a, 1]
    # requires the following permutation
    # [1, 2, 0, 5, 3, 4, 7, 8, 6, 9]

    # Permute coefficients in C
    # [c^2, a^2, b^2, ca, cb, ab, c, a, b, 1]
    # requires the following permutation
    # [2, 0, 1, 5, 7, 4, 9, 6, 8, 3]

    # We need to iterate over all this permutations to find the most stable one
    perms = [
        list(range(10)),
        [1, 2, 0, 5, 3, 4, 7, 8, 6, 9],
        [2, 0, 1, 4, 5, 3, 8, 6, 7, 9],
    ]

    min_cond = np.float("+inf")
    min_idx = None

    for i, perm in enumerate(perms):
        # since we need to access columns, it's more convenient to keep it transposed
        A_r = (A.T)[perm]

        # If A_r is formed by ten cols [a_0, ..., a_9]
        # H is the concat of cols [a_1, a_2, a_5]
        H = A_r[[1, 2, 5]].T
        cond = np.linalg.cond(H)
        if cond < min_cond:
            min_cond = cond
            min_idx = i

    # Pick the final permutation
    if min_idx is None:
        raise ValueError("Matrix A is ill-conditioned")
    A_r = (A.T)[perms[min_idx]]
    H = A_r[[1, 2, 5]].T

    # This should not be needed anymore
    # # Check if the matrix is invertible
    # if np.linalg.matrix_rank(H) < 3:
    #     raise NotImplementedError

    H_inv = np.linalg.inv(H)

    # Observe the eq system
    # [b^2, c^2, bc]^T = H^-1 * [p1, ..., p3] [b, c, 1]^T
    # and let's denote H^-1 * [p1, ..., p3] as J
    # J is composed of three columns [j1(a), j2(a), j3(a)]
    # j1 = j11 * a + j10
    # j2 = j21 * a + j20
    # j3 = j32 * a^2 + j31 * a + j30
    j10 = -H_inv @ A_r[7]
    j11 = -H_inv @ A_r[3]  # *a
    j20 = -H_inv @ A_r[8]
    j21 = -H_inv @ A_r[4]  # *a
    j30 = -H_inv @ A_r[9]
    j31 = -H_inv @ A_r[6]  # *a
    j32 = -H_inv @ A_r[0]  # *a^2

    # Spread cols to individual coeffs
    j110, j210, j310 = j10
    j111, j211, j311 = j11
    j120, j220, j320 = j20
    j121, j221, j321 = j21
    j130, j230, j330 = j30
    j131, j231, j331 = j31
    j132, j232, j332 = j32

    #

    # J = Matrix([
    #  ...:     [j11, j12, j13],
    #  ...:     [j21, j22, j23],
    #  ...:     [j31, j32, j33],
    #  ...: ])

    # Apply the identities, perform double substitution and it will yield an homogeneous
    # system of 3 equations, with respect to [b, c, 1]. Each component of that matrix
    # will be a polynomial in a. Compute the determinant expression and it will
    # yield an 8th degree polynomial, with the coefficients bellow

    # This has an error from constructing the identities. A copy paste mistake
    # Computing the coefficients yields some very big expressions which python's ast
    # cannot handle. We send this task to plain old C.
    # fmt: off
    coeffs = coefficients((
        j110, j210, j310,
        j111, j211, j311,
        j120, j220, j320,
        j121, j221, j321,
        j130, j230, j330,
        j131, j231, j331,
        j132, j232, j332,
    ))
    # fmt: on

    # coeffs = (c8, c7, c6, c5, c4, c3, c2, c1, c0))
    roots = np.roots(coeffs)

    # filter out roots
    a = np.real(roots) if allow_imag_roots else np.real(roots[np.isreal(roots)])

    # In here we actually show how to build the homogeneous system from before
    # now that we know a

    # Things are different here from the old expressions
    # seems there was also an error here

    # fmt: off

    # Applying the identities
    # 1 -
    # expand(b^2*c - bcb)

    # just dep on a
    m11 = a**3*(j111*j332 + j121*j232 - j132*j311 - j321*j332) \
        + a**2*(j110*j332 + j111*j331 + j120*j232 + j121*j231 - j131*j311 - j132*j310 - j320*j332 - j321*j331) \
        + a*(j110*j331 + j111*j330 + j120*j231 + j121*j230 - j130*j311 - j131*j310 - j320*j331 - j321*j330) \
        + j110*j330 + j120*j230 - j130*j310 - j320*j330

    # dep on a and b
    m12 = a**2*(j121*j211 - j311*j321 - j332) \
        + a*(j120*j211 + j121*j210 - j310*j321 - j311*j320 - j331) \
        + j120*j210 - j310*j320 - j330

    # dep on a and c
    m13 = a**2*(j111*j321 + j121*j221 - j121*j311 + j132 - j321**2) \
        + a*(j110*j321 + j111*j320 + j120*j221 - j120*j311 + j121*j220 - j121*j310 + j131 - 2*j320*j321) \
        + j110*j320 + j120*j220 - j120*j310 + j130 - j320**2

    # 2 -
    # expand(bcc - c^2b)

    # scalar term
    m21 = a**3*(-j132*j211 - j221*j332 + j232*j321 + j311*j332) \
        + a**2*(-j131*j211 - j132*j210 - j220*j332 - j221*j331 + j231*j321 + j232*j320 + j310*j332 + j311*j331) \
        + a*(-j130*j211 - j131*j210 - j220*j331 - j221*j330 + j230*j321 + j231*j320 + j310*j331 + j311*j330) \
        - j130*j210 - j220*j330 + j230*j320 + j310*j330

    # term in b
    m22 = a**2*(-j111*j211 + j211*j321 - j221*j311 - j232 + j311**2) \
        + a*(-j110*j211 - j111*j210 + j210*j321 + j211*j320 - j220*j311 - j221*j310 - j231 + 2*j310*j311) \
        - j110*j210 + j210*j320 - j220*j310 - j230 + j310**2

    # term in c
    m23 = a**2*(-j121*j211 + j311*j321 + j332) \
        + a*(-j120*j211 - j121*j210 + j310*j321 + j311*j320 + j331) \
        - j120*j210 + j310*j320 + j330

    # 3 -
    # expand(J[2,:]*Matrix([[b2, bc, b],[bc, c2, c],[b, c, 1]])*(J[2,:].T))

    # 1
    m31 = a**4*(-j111*j132*j211 - j111*j221*j332 - j121*j211*j332 - j121*j221*j232 \
            - j132*j232 + j132*j311**2 + j232*j321**2 + 2*j311*j321*j332 + j332**2) \
        + a**3*(-j110*j132*j211 - j110*j221*j332 - j111*j131*j211 - j111*j132*j210 \
            - j111*j220*j332 - j111*j221*j331 - j120*j211*j332 - j120*j221*j232 \
            - j121*j210*j332 - j121*j211*j331 - j121*j220*j232 - j121*j221*j231 \
            - j131*j232 + j131*j311**2 - j132*j231 + 2*j132*j310*j311 + j231*j321**2 \
            + 2*j232*j320*j321 + 2*j310*j321*j332 + 2*j311*j320*j332 + 2*j311*j321*j331 + 2*j331*j332) \
        + a**2*(-j110*j131*j211 - j110*j132*j210 - j110*j220*j332 - j110*j221*j331 \
            - j111*j130*j211 - j111*j131*j210 - j111*j220*j331 - j111*j221*j330 \
            - j120*j210*j332 - j120*j211*j331 - j120*j220*j232 - j120*j221*j231 \
            - j121*j210*j331 - j121*j211*j330 - j121*j220*j231 - j121*j221*j230 \
            - j130*j232 + j130*j311**2 - j131*j231 + 2*j131*j310*j311 - j132*j230 \
            + j132*j310**2 + j230*j321**2 + 2*j231*j320*j321 + j232*j320**2 + 2*j310*j320*j332 \
            + 2*j310*j321*j331 + 2*j311*j320*j331 + 2*j311*j321*j330 + 2*j330*j332 + j331**2) \
        + a*(-j110*j130*j211 - j110*j131*j210 - j110*j220*j331 - j110*j221*j330 - j111*j130*j210 \
            - j111*j220*j330 - j120*j210*j331 - j120*j211*j330 - j120*j220*j231 - j120*j221*j230 \
            - j121*j210*j330 - j121*j220*j230 - j130*j231 + 2*j130*j310*j311 - j131*j230 \
            + j131*j310**2 + 2*j230*j320*j321 + j231*j320**2 + 2*j310*j320*j331 + 2*j310*j321*j330 \
            + 2*j311*j320*j330 + 2*j330*j331) \
        - j110*j130*j210 - j110*j220*j330 - j120*j210*j330 - j120*j220*j230 - j130*j230 \
        + j130*j310**2 + j230*j320**2 + 2*j310*j320*j330 + j330**2

    # b
    m32 = a**3*(-j111**2*j211 - j111*j221*j311 - j111*j232 + j111*j311**2 \
            - j121*j211*j221 - j121*j211*j311 - j132*j211 + j211*j321**2 + 2*j311**2*j321 + 2*j311*j332) \
        + a**2*(-2*j110*j111*j211 - j110*j221*j311 - j110*j232 + j110*j311**2 \
            - j111**2*j210 - j111*j220*j311 - j111*j221*j310 - j111*j231 + 2*j111*j310*j311 \
            - j120*j211*j221 - j120*j211*j311 - j121*j210*j221 - j121*j210*j311 - j121*j211*j220 \
            - j121*j211*j310 - j131*j211 - j132*j210 + j210*j321**2 + 2*j211*j320*j321 \
            + 4*j310*j311*j321 + 2*j310*j332 + 2*j311**2*j320 + 2*j311*j331) \
        + a*(-j110**2*j211 - 2*j110*j111*j210 - j110*j220*j311 - j110*j221*j310 - j110*j231 \
            + 2*j110*j310*j311 - j111*j220*j310 - j111*j230 + j111*j310**2 - j120*j210*j221 \
            - j120*j210*j311 - j120*j211*j220 - j120*j211*j310 - j121*j210*j220 - j121*j210*j310 \
            - j130*j211 - j131*j210 + 2*j210*j320*j321 + j211*j320**2 + 2*j310**2*j321 \
            + 4*j310*j311*j320 + 2*j310*j331 + 2*j311*j330) \
        - j110**2*j210 - j110*j220*j310 - j110*j230 + j110*j310**2 - j120*j210*j220 \
        - j120*j210*j310 - j130*j210 + j210*j320**2 + 2*j310**2*j320 + 2*j310*j330

    # c
    m33 = a**3*(-j111*j121*j211 - j111*j221*j321 - j121*j211*j321 - j121*j221**2 \
            - j121*j232 + j121*j311**2 - j132*j221 + j221*j321**2 + 2*j311*j321**2 + 2*j321*j332) \
        + a**2*(-j110*j121*j211 - j110*j221*j321 - j111*j120*j211 - j111*j121*j210 \
            - j111*j220*j321 - j111*j221*j320 - j120*j211*j321 - j120*j221**2 - j120*j232 \
            + j120*j311**2 - j121*j210*j321 - j121*j211*j320 - 2*j121*j220*j221 - j121*j231 \
            + 2*j121*j310*j311 - j131*j221 - j132*j220 + j220*j321**2 + 2*j221*j320*j321 \
            + 2*j310*j321**2 + 4*j311*j320*j321 + 2*j320*j332 + 2*j321*j331) \
        + a*(-j110*j120*j211 - j110*j121*j210 - j110*j220*j321 - j110*j221*j320 - j111*j120*j210 \
            - j111*j220*j320 - j120*j210*j321 - j120*j211*j320 - 2*j120*j220*j221 - j120*j231 \
            + 2*j120*j310*j311 - j121*j210*j320 - j121*j220**2 - j121*j230 + j121*j310**2 \
            - j130*j221 - j131*j220 + 2*j220*j320*j321 + j221*j320**2 + 4*j310*j320*j321 \
            + 2*j311*j320**2 + 2*j320*j331 + 2*j321*j330) \
        - j110*j120*j210 - j110*j220*j320 - j120*j210*j320 - j120*j220**2 - j120*j230 \
        + j120*j310**2 - j130*j220 + j220*j320**2 + 2*j310*j320**2 + 2*j320*j330

    # fmt: on

    # Write M as
    # M = [[m11, m12, m13], [m21, m22, m23], [m31, m32, 33]]
    #
    # Beware that in this system the "unknown" order is [1, b, c]
    M = np.block(
        [
            [m11[:, None, None], m12[:, None, None], m13[:, None, None]],
            [m21[:, None, None], m22[:, None, None], m23[:, None, None]],
            [m31[:, None, None], m32[:, None, None], m33[:, None, None]],
        ]
    )

    # To find b, c from a we solve our linear system. We iterate over all solutions
    # under the intuition that running an SVD (least sqrs) on 8x3 matrix, which doesn't
    # exploit sparsity is worse than iterating over each equation system individually
    bc = np.empty((len(M), 2))
    for i in range(len(M)):
        bc[i], _, _, _ = np.linalg.lstsq(M[i, :, 1:], -M[i, :, 0], rcond=None)

    b, c = bc.T

    # Revert a, b, c swapping
    a, b, c = np.roll((a, b, c), min_idx, axis=0)
    return a, b, c
