import unittest

import sympy as sp

from pysplines.bsplines import CoreBspline
from pysplines.alexpression import ALexpression


class TestCoreBspline(unittest.TestCase):
    def setUp(self):
        cv = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]
        self.core_bspline = CoreBspline(cv, n=120)

    def test_bspline_domain(self):
        self.assertEqual(self.core_bspline.n, 120)
        self.assertEqual(len(self.core_bspline.dom), 120)

    def test_bspline_basis(self):
        x = sp.var("x")
        analytical_basis_expressions = [
            sp.Piecewise(
                (-(x ** 3) + 3 * x ** 2 - 3 * x + 1, (x >= 0) & (x <= 1)), (0, True)
            ),
            sp.Piecewise(
                (7 * x ** 3 / 4 - 9 * x ** 2 / 2 + 3 * x, (x >= 0) & (x <= 1)),
                (-(x ** 3) / 4 + 3 * x ** 2 / 2 - 3 * x + 2, (x >= 1) & (x <= 2)),
                (0, True),
            ),
            sp.Piecewise(
                (-(x ** 3) + 3 * x ** 2 / 2, (x >= 0) & (x <= 1)),
                (x ** 3 - 9 * x ** 2 / 2 + 6 * x - 2, (x >= 1) & (x <= 2)),
                (0, True),
            ),
            sp.Piecewise(
                (x ** 3 / 4, (x >= 0) & (x <= 1)),
                (-7 * x ** 3 / 4 + 6 * x ** 2 - 6 * x + 2, (x >= 1) & (x <= 2)),
                (0, True),
            ),
            sp.Piecewise(
                (x ** 3 - 3 * x ** 2 + 3 * x - 1, (x >= 1) & (x <= 2)), (0, True)
            ),
        ]

        for actual_basis, expected_basis in zip(
            self.core_bspline.bspline_basis, analytical_basis_expressions
        ):
            self.assertEqual(actual_basis.aform, expected_basis)

    def test_bspline_expression(self):
        self.assertEqual(len(self.core_bspline.bspline), 2)
        for bspline in self.core_bspline.bspline:
            self.assertIsInstance(bspline, ALexpression)

    def test_bspline_evaluate_expression(self):
        expression = self.core_bspline.bspline

        self.assertEqual(len(self.core_bspline.evaluate_expression(expression)), 120)
        self.assertListEqual(
            self.core_bspline.evaluate_expression(expression, domain=1), [1.0, 1.0]
        )
        self.assertListEqual(
            self.core_bspline.evaluate_expression(expression, [1]), [1.0, 1.0]
        )
        self.assertListEqual(
            self.core_bspline.evaluate_expression(expression, [1, 2]),
            [[1.0, 1.0], [2.0, 2.0]],
        )
