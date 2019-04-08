"""
Create and plot and example periodic B-spline with given control points cv and of degree 3.

>>> cv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.5], [0.0, 1.0]]
"""
from pysplines.bsplines import Bspline


def plot_periodic_bspline():
    control_points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.5], [0.0, 1.0]]
    periodic_bspline = Bspline(control_points, periodic=True)
    periodic_bspline.plot()


if __name__ == "__main__":
    plot_periodic_bspline()
