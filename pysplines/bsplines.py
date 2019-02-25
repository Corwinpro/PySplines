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
        self.n = n

        self.max_param = self.cv.shape[0] - (self.degree * (1 - self.periodic))
        self.kv = self.kv_bspline()
        self.t_to_point_dict = dict()

        self.bspline_basis = self.construct_bspline_basis()
        self.spline = self.construct_bspline_expression()
        self.dom = np.linspace(0, self.max_param, self.n)
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
        bspline_expression = [0, 0]

        for i in range(len(self.cv)):
            bspline_expression[0] += self.cv[i][0] * self.bspline_basis[i].aform
            bspline_expression[1] += self.cv[i][1] * self.bspline_basis[i].aform
        return ALexpression(bspline_expression)

    def get_displacement_from_point(self, point, controlPointNumber):
        raise NotImplementedError
        t = self.get_t_from_point(point)
        displacement = self.bspline_basis_lambda[controlPointNumber].lform(t).item()
        return [displacement, displacement]

    def bspline_getSurface(self):

        self.rvals = self.evaluate_expression(self.spline)

        for i in range(len(self.rvals)):
            self.t_to_point_dict[self.dom[i]] = self.rvals[i]
            self.rvals[i][0] = math.trunc(self.rvals[i][0] * 1.0e8) / 1.0e8
            self.rvals[i][1] = math.trunc(self.rvals[i][1] * 1.0e8) / 1.0e8

    def evaluate_expression(self, expression, point=None):
        """
        Given a sympy expression, a point (or set of points), calculates
        values of the expression at the point(s).
        """
        expression_val = []
        if isinstance(expression, list):
            n = len(expression)
            if point is not None:
                vals = []
                for i in range(n):
                    vals.append(float(expression[i].subs(self.x, point)))
                return vals
            else:
                lambda_expression = [
                    sympy.lambdify(self.x, expression[i]) for i in range(n)
                ]
                for r in self.dom:
                    vals = []
                    for i in range(n):
                        vals.append(lambda_expression[i](r))
                    expression_val.append(vals)
        else:
            if point is not None:
                return float(expression.subs(self.x, point))
            else:
                lambda_expression = sympy.lambdify(self.x, expression)
                for r in self.dom:
                    val = lambda_expression(r)
                    expression_val.append(float(val))

        return expression_val

    def plot(self, linetype="-", window=plt):

        window.plot(
            np.array(self.rvals)[:, 0],
            np.array(self.rvals)[:, 1],
            linetype,
            color="black",
        )
        # self.plot_cv(window)
        window.show()

