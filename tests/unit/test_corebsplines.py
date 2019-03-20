import pytest
from pysplines.bsplines import CoreBspline

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

