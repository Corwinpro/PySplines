import unittest

import numpy as np

from pysplines.bsplines import Bspline

_TOLERANCE = 5.0e-7


class TestBspline(unittest.TestCase):
    def setUp(self):
        cv = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]
        self.bspline = Bspline(cv, n=120)

    def test_edge_points(self):
        rvals = np.array(self.bspline.rvals)
        control_points = self.bspline.cv
        self.assertTrue(np.linalg.norm(rvals[0] - control_points[0]) < _TOLERANCE)
        self.assertTrue(np.linalg.norm(rvals[-1] - control_points[-1]) < _TOLERANCE)

        normal_vector_left = self.bspline.normal(control_points[0])
        normal_vector_left_expected = np.array([-1.0, 0.0])
        normal_vector_right = self.bspline.normal(control_points[0])
        normal_vector_right_expected = np.array([-1.0, 0.0])

        self.assertTrue(
            np.linalg.norm(normal_vector_left - normal_vector_left_expected)
            < _TOLERANCE
        )
        self.assertTrue(
            (
                np.linalg.norm(normal_vector_right - normal_vector_right_expected)
                < _TOLERANCE
            )
        )

    def test_get_t_from_point(self):
        self.assertTrue(
            abs(self.bspline.get_t_from_point(self.bspline.cv[0]) - self.bspline.kv[0])
            < _TOLERANCE
        )
        self.assertTrue(
            abs(
                self.bspline.get_t_from_point(self.bspline.cv[2])
                - self.bspline.kv[-1] / 2.0
            )
            < _TOLERANCE
        )
        self.assertTrue(
            abs(
                self.bspline.get_t_from_point(self.bspline.cv[-1]) - self.bspline.kv[-1]
            )
            < _TOLERANCE
        )
