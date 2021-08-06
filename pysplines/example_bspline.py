from pysplines.bsplines import Bspline


class Example_BSpline(Bspline):
    """
    Example BSpline with degree d = 2, n = 100 plotted points, default periodic = False
    """

    def __init__(self, degree=3, n=100, periodic=False, **kwargs):
        control_points = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, -0.1],
            [0.3, 0.2],
            [0.4, 0.0],
            [0.5, 0.1],
        ]
        self.example_cv = control_points
        super().__init__(
            self.example_cv, degree=degree, n=n, periodic=periodic, **kwargs
        )
