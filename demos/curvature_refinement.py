"""
Example of curvature refinement procedure.
We compare a coarse B-spilne (without refinement) to an adapted B-spline,
with the second one being much smoother in the areas with high curvature.

>>>    bspline = Example_BSpline(n=100, refine=False)
>>>    bspline.plot(linetype="o-", color="red", show=False)

>>>    bspline = Example_BSpline(n=100, refine=True)
>>>    bspline.plot(linetype="*-")
"""

from pysplines.example_bspline import Example_BSpline

if __name__ == "__main__":

    bspline = Example_BSpline(n=100, refine=False)
    bspline.plot(linetype="o-", color="red", show=False)

    bspline = Example_BSpline(n=100, refine=True)
    bspline.plot(linetype="*-")
