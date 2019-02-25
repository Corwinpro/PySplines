import numpy as np
import sympy
from scipy.integrate import simps


class sympy_Bspline:
    def __init__(self, cv, degree=3, n=100, periodic=False):
        """
            Clamped B-Spline with sympy

            cv: control points vector
            degree:   Curve degree
            n: N discretization points
            periodic: default - False; True - Curve is closed

            kv: bspline knot vector
        """
        self.x = sympy.var("x")

        self.cv = np.array(cv)

        self.periodic = periodic
        self.degree = degree
        self.max_param = self.cv.shape[0] - (self.degree * (1 - self.periodic))
        self.kv = self.kv_bspline()

        self.construct_bspline_basis()

        self.spline = self.sympy_bspline_expression()

    def kv_bspline(self):
        """ 
            Calculates knot vector of a bspline
        """
        cv = np.asarray(self.cv)
        count = cv.shape[0]

        # Closed curve
        if self.periodic:
            kv = np.arange(-self.degree, count + self.degree + 1)
            factor, fraction = divmod(count + self.degree + 1, count)
            cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)), -1, axis=0)
            cv = cv[:-1]
            degree = np.clip(self.degree, 1, self.degree)
        # Opened curve
        else:
            degree = np.clip(self.degree, 1, count - 1)
            kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)

        self.cv = cv
        return list(kv)

    def construct_bspline_basis(self):
        self.bspline_basis = []
        for i in range(len(self.cv)):
            self.bspline_basis.append(
                sympy.bspline_basis(self.degree, self.kv, i, self.x)
            )
        self.lambdify_bspline_basis_expressions()

    def lambdify_bspline_basis_expressions(self):

        self.bspline_basis_lambda = []
        for i in range(len(self.cv)):
            self.bspline_basis_lambda.append(
                sympy.lambdify(self.x, self.bspline_basis[i])
            )

    def sympy_bspline_expression(self):
        bspline_expression = [0, 0]

        for i in range(len(self.cv)):
            bspline_expression[0] += self.cv[i][0] * self.bspline_basis[i]
            bspline_expression[1] += self.cv[i][1] * self.bspline_basis[i]
        return bspline_expression
