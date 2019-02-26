from pysplines.example_bspline import Example_BSpline
import matplotlib.pyplot as plt

if __name__ == "__main__":

    bspline = Example_BSpline(n=100, refine=False)
    bspline.plot(linetype="o-", color="red", show=False)

    bspline = Example_BSpline(n=100, refine=True)
    bspline.plot(linetype="o-")
