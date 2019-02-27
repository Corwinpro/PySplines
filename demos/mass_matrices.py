"""
Illustrates the mass matrix of an Example_Bspline with 6 basis functions.
Use bspline.mass_matrix(DLMM=True) to get the diagonal only elements of the mass matrix.
"""

from pysplines.example_bspline import Example_BSpline

if __name__ == "__main__":

    bspline = Example_BSpline(n=10)
    matrix = bspline.mass_matrix()

    import matplotlib.pyplot as plt

    plt.matshow(matrix, cmap=plt.cm.Blues)
    plt.title("Bsplines Mass Matrix", fontsize=12)
    plt.colorbar()
    plt.show()
