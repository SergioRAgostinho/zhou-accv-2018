import unittest
import numpy as np
from zhou_accv_2018 import e3q3


class E3Q3(unittest.TestCase):
    def test_ill_conditioned_input(self):
        # too many coefficients are set to zero
        A = np.array(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, -4],
                [1, 1, 1, 0, 0, 0, -4, 0, 0, 3],
                [1, 1, 1, 0, 0, 0, -3.5, 0, 0, 2.125],
            ]
        )
        self.assertRaises(ValueError, e3q3, A)


if __name__ == "__main__":
    unittest.main()
