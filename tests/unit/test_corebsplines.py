import pytest
from pysplines.bsplines import CoreBspline
from pysplines.alexpression import ALexpression
import sympy as sp

TEST_N = 120


@pytest.fixture
def bspline_core():
    cv = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]
    bspline = CoreBspline(cv, n=TEST_N)
    return bspline


def test_bspline_domain(bspline_core):
    bspline = bspline_core
    assert bspline.n == TEST_N
    assert len(bspline.dom) == TEST_N


def test_bspline_basis(bspline_core):
    bspline = bspline_core
    x = sp.var("x")
    analytical_basis = [
        sp.Piecewise(
            (-x ** 3 + 3 * x ** 2 - 3 * x + 1, (x >= 0) & (x <= 1)), (0, True)
        ),
        sp.Piecewise(
            (7 * x ** 3 / 4 - 9 * x ** 2 / 2 + 3 * x, (x >= 0) & (x <= 1)),
            (-x ** 3 / 4 + 3 * x ** 2 / 2 - 3 * x + 2, (x >= 1) & (x <= 2)),
            (0, True),
        ),
        sp.Piecewise(
            (-x ** 3 + 3 * x ** 2 / 2, (x >= 0) & (x <= 1)),
            (x ** 3 - 9 * x ** 2 / 2 + 6 * x - 2, (x >= 1) & (x <= 2)),
            (0, True),
        ),
        sp.Piecewise(
            (x ** 3 / 4, (x >= 0) & (x <= 1)),
            (-7 * x ** 3 / 4 + 6 * x ** 2 - 6 * x + 2, (x >= 1) & (x <= 2)),
            (0, True),
        ),
        sp.Piecewise((x ** 3 - 3 * x ** 2 + 3 * x - 1, (x >= 1) & (x <= 2)), (0, True)),
    ]

    for i, basis in enumerate(bspline.bspline_basis):
        assert basis.aform == analytical_basis[i]


def test_bspline_expression(bspline_core):
    bspline = bspline_core
    assert len(bspline.bspline) == 2
    for i in range(len(bspline.bspline)):
        assert isinstance(bspline.bspline[i], ALexpression)


def test_bspline_evaluate_expression(bspline_core):
    bspline = bspline_core
    expression = bspline.bspline

    assert len(bspline.evaluate_expression(expression)) == TEST_N
    assert bspline.evaluate_expression(expression, domain=1) == [1.0, 1.0]
    assert bspline.evaluate_expression(expression, [1]) == [1.0, 1.0]
    assert bspline.evaluate_expression(expression, [1, 2]) == [[1.0, 1.0], [2.0, 2.0]]

