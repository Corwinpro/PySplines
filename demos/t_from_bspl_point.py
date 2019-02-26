from pysplines.example_bspline import Example_BSpline
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    bspline = Example_BSpline(n = 100)
    trial_point = (0.00601469, 0.005876951)
    plt.plot(*trial_point, '*')
    bspline.plot(linetype='o-')
    #print(bspline.point_to_t_dict)
    t_predict = bspline.get_t_from_point(trial_point)
    print(t_predict)