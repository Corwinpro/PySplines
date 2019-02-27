"""
We predict the internal parameter value of a B-spline near a trial_point.
"""

from pysplines.example_bspline import Example_BSpline
import matplotlib.pyplot as plt

if __name__ == "__main__":

    bspline = Example_BSpline(n=100)
    trial_point = (
        (0.0523811252668318 + 0.052628073457715752) / 2.0,
        (0.037611544608820317 + 0.037713874638076669) / 2.0,
    )
    plt.plot(*trial_point, "*")
    bspline.plot(linetype="o-")
    # print(bspline.point_to_t_dict)
    t_predict = bspline.get_t_from_point(trial_point)
    print(t_predict)
