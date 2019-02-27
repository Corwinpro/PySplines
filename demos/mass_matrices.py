from pysplines.example_bspline import Example_BSpline

if __name__ == "__main__":

    bspline = Example_BSpline(n=10)
    matrix = bspline.mass_matrix()

    import matplotlib.pyplot as plt
    plt.matshow(matrix, cmap=plt.cm.Blues)
    plt.title("Bsplines Mass Matrix", fontsize=12)
    plt.colorbar()
    plt.show()