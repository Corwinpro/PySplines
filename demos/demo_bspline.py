"""
Basic usage of pysplines
Create and plot and example B-spline with given control points cv and of degree 3.

>>> cv = [
>>>    [0.0, 0.0],
>>>    [0.1, 0.1],
>>>    [0.2, -0.1],
>>>    [0.3, 0.2],
>>>    [0.4, 0.0],
>>>    [0.5, 0.1]
>>> ]
"""
from pysplines.example_bspline import Example_BSpline


def create_bspline():
    example_bspline = Example_BSpline()
    example_bspline.plot()


if __name__ == "__main__":
    create_bspline()
