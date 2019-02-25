import numpy as np
import sympy
import math
from scipy.integrate import simps
import matplotlib.pyplot as plt
import warnings


class ALexpression:
    """
    A dual Analytical-Lambdified form expression of a single variable

    Attributes:
    - aform: analytical sympy form of the expression
    - lform: fully lambdified for of the expression
    """

    def __init__(self, sympy_expression):
        self.aform = sympy_expression
        self.t = sympy_expression.free_symbols
        if len(self.t) > 1:
            warnings.warn(
                "Be careful with the lambdified expression {}, as it has more than one free symbol.".format(
                    self.aform
                )
            )

        self.lform = sympy.lambdify(self.t, self.aform)


class sympy_Bspline:
    def __init__(self, cv, degree=3, n=100, periodic=False):
        """
            Clamped B-Spline with sympy

            space_dimension: problem dimension (2D only)
            cv: control points vector
            degree:   Curve degree
            n: N discretization points
            periodic: default - False; True - Curve is closed

            kv: bspline knot vector
        """
        self.x = sympy.var("x")

        self.space_dimension = 2
        self.cv = np.array(cv)
        self.periodic = periodic
        self.degree = degree
        self.n = n

        self.max_param = self.cv.shape[0] - (self.degree * (1 - self.periodic))
        self.kv = self.kv_bspline()
        self.dom = np.linspace(0, self.max_param, self.n)

        self.t_to_point_dict = dict()

        self.bspline_basis = self.construct_bspline_basis()
        self.bspline = self.construct_bspline_expression()
        self.bspline_getSurface()

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
        bspline_basis = []
        for i in range(len(self.cv)):
            bspline_basis.append(
                ALexpression(sympy.bspline_basis(self.degree, self.kv, i, self.x))
            )
        return bspline_basis

    def construct_bspline_expression(self):
        bspline_expression = [0] * self.space_dimension

        for i in range(len(self.cv)):
            for j in range(self.space_dimension):
                bspline_expression[j] += self.cv[i][j] * self.bspline_basis[i].aform
        return [
            ALexpression(bspline_expression[i]) for i in range(self.space_dimension)
        ]

    def get_displacement_from_point(self, point, controlPointNumber):
        raise NotImplementedError
        t = self.get_t_from_point(point)
        displacement = self.bspline_basis[controlPointNumber].lform(t).item()
        return [displacement, displacement]

    def bspline_getSurface(self):

        self.rvals = self.evaluate_expression(self.bspline)

        for i in range(len(self.rvals)):
            self.t_to_point_dict[self.dom[i]] = self.rvals[i]
            for j in range(self.space_dimension):
                self.rvals[i][j] = math.trunc(self.rvals[i][j] * 1.0e8) / 1.0e8

    def evaluate_expression(self, expression, point=None):
        """
        Given a sympy expression, a point (or set of points), calculates
        values of the expression at the point(s).

        Returns: a list of the expression values

        TODO: check all possible usage cases
        """
        if point is None:
            domain = self.dom
        else:
            domain = (point,)

        expression_val = []
        if isinstance(expression, list):
            n = len(expression)
            for r in domain:
                val = [float(expression[i].lform(r)) for i in range(n)]
                expression_val.append(val)
        else:
            for r in domain:
                val = expression.lform(r)
                expression_val.append(val)

        return expression_val

    def plot(self, linetype="-", window=plt):

        window.plot(
            np.array(self.rvals)[:, 0],
            np.array(self.rvals)[:, 1],
            linetype,
            color="black",
        )
        # self.plot_cv(window)
        # window.show()

