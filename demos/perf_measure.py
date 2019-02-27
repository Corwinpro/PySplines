"""
This demo illustrates the performance of a simple B-spline constructor.
The bottleneck of the performance is the bspline_basis method, which
solely relies on the sympy bspline basis generating method.
"""

from pysplines.example_bspline import Example_BSpline
from pysplines.decorators import timethis


if __name__ == "__main__":

    @timethis(n_iter=10)
    def create_bspline():
        example_bspline = Example_BSpline()

    @timethis(n_iter=10)
    def create_basis(bspline):
        """
        We can see that the most expensive part of the bspline construction
        is creating of the basis symbolic expression
        """
        bspline.construct_bspline_basis()

    create_bspline()

    bspline = Example_BSpline()
    create_basis(bspline)
