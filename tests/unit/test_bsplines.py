import pytest
from pysplines.bsplines import Bspline
from pysplines.alexpression import ALexpression
import sympy as sp
import numpy as np

TEST_N = 120
_TOLERANCE = 5.0e-7


@pytest.fixture
def default_bspline():
    cv = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]
    bspline = Bspline(cv, n=TEST_N)
    return bspline


def test_edge_points(default_bspline):
    rvals = np.array(default_bspline.rvals)
    control_points = default_bspline.cv
    assert np.linalg.norm(rvals[0] - control_points[0]) < _TOLERANCE
    assert np.linalg.norm(rvals[-1] - control_points[-1]) < _TOLERANCE

    normal_vector_left = default_bspline.normal(control_points[0])
    normal_vector_left_expected = np.array([-1.0, 0.0])
    normal_vector_right = default_bspline.normal(control_points[0])
    normal_vector_right_expected = np.array([-1.0, 0.0])

    assert np.linalg.norm(normal_vector_left - normal_vector_left_expected) < _TOLERANCE
    assert (
        np.linalg.norm(normal_vector_right - normal_vector_right_expected) < _TOLERANCE
    )


def test_get_t_from_point(default_bspline):
    assert (
        abs(
            default_bspline.get_t_from_point(default_bspline.cv[0])
            - default_bspline.kv[0]
        )
        < _TOLERANCE
    )
    assert (
        abs(
            default_bspline.get_t_from_point(default_bspline.cv[2])
            - default_bspline.kv[-1] / 2.0
        )
        < _TOLERANCE
    )
    assert (
        abs(
            default_bspline.get_t_from_point(default_bspline.cv[-1])
            - default_bspline.kv[-1]
        )
        < _TOLERANCE
    )


def test_bspline_evaluate_expression(default_bspline):
    pass

