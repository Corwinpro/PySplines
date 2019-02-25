from pysplines.bsplines import sympy_Bspline
from pysplines.timeperformance import timethis

class Example_BSpline(sympy_Bspline):
    """
		Example BSpline with degree d = 3, n = 100 plotted points, default periodic = False
	"""

    def __init__(self, degree=2, n=100, periodic=False):
        top_control_points = [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.4, 0.0],
        ]
        self.example_cv = top_control_points
        sympy_Bspline.__init__(
            self, self.example_cv, degree=degree, n=n, periodic=periodic
        )


if __name__ == "__main__":

    @timethis(n_iter=5)
    def create_bspline():
        example_bspline = Example_BSpline()

    create_bspline()
