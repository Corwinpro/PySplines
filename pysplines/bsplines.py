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

    def __getitem__(self, t):
        return self.lform(t)


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

        self.point_to_t_dict = dict()
        self.tolerance = 1.0e-8

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

    def bspline_getSurface(self):

        self.rvals = self.evaluate_expression(self.bspline)

        for i in range(len(self.rvals)):
            self.point_to_t_dict[tuple(self.rvals[i])] = self.dom[i]
            for j in range(self.space_dimension):
                self.rvals[i][j] = (
                    math.trunc(self.rvals[i][j] / self.tolerance) * self.tolerance
                )

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
                val = [expression[i][r] for i in range(n)]
                expression_val.append(val)
        elif isinstance(expression, ALexpression):
            for r in domain:
                val = expression[r]
                expression_val.append(val)
        else:
            raise NotImplementedError

        return expression_val

    def plot(self, linetype="-", window=plt, **kwargs):

        window.plot(
            np.array(self.rvals)[:, 0],
            np.array(self.rvals)[:, 1],
            linetype,
            color="black",
            **kwargs
        )
        # self.plot_cv(window)
        window.show()

    def get_displacement_from_point(self, point, controlPointNumber):
        t = self.get_t_from_point(point)
        displacement = self.bspline_basis[controlPointNumber][t]
        return [displacement, displacement]

    def get_t_from_point(self, point):
        """
        Given a point, we check if the point lies on the bspline, and what is 
        the internal parameter 't' it corresponds to.

        TODO:
        - Now we only return t for points that are already on the existing lines.
          However, this is not always correct - if the bspline discretization is 
          too coarse, some of the real points won't be on the existing lines.
          We need to check if the 'point' is actually near the point, predicted by 't'.
        """
        point = tuple(point)
        if point in self.point_to_t_dict:
            return self.point_to_t_dict[point]

        min_dist = previous_distance = 1.0 / self.tolerance

        for r in self.rvals:
            current_distance = np.linalg.norm(np.array(r) - np.array(point))
            if current_distance > previous_distance:
                continue
            if current_distance < min_dist:
                min_dist = current_distance
                closest_point = r
            previous_distance = current_distance

        index = self.rvals.index(closest_point)
        t = self.dom[index]

        # If we are 'very' close to an existing point, we just return t
        # This helps if a point is 'out of the domain' but still very close
        if min_dist < self.tolerance:
            return t
            # We get the distance to the points to the left and to the right of the closest point
            # ...---o---o---left---closest---right---o----o---...
            # If the first point of the spline is the nearest, there are no points to the left
            # And the second nearest point (to the right) has the second shortest distance
            # points_distance is the distance between to bspline points closest to the point
            # closest---right---o----o---...
        if index == 0:
            second_distance = np.linalg.norm(np.array(self.rvals[1]) - np.array(point))
            points_distance = np.linalg.norm(
                np.array(self.rvals[1]) - np.array(self.rvals[0])
            )
            second_t = self.dom[1]
            # Same happens if the closest point is the last one on the spline
            # ...---o---left---closest.
        elif index == len(self.dom) - 1:
            second_distance = np.linalg.norm(np.array(self.rvals[-2]) - np.array(point))
            points_distance = np.linalg.norm(
                np.array(self.rvals[-1]) - np.array(self.rvals[-2])
            )
            second_t = self.dom[-2]
            # Otherwise, the closest bspline point to the point is somewhere on the curve
        else:
            left_point_distance = np.linalg.norm(
                np.array(self.rvals[index - 1]) - np.array(point)
            )
            right_point_distance = np.linalg.norm(
                np.array(self.rvals[index + 1]) - np.array(point)
            )

            r_right = [
                self.rvals[index + 1][0] - point[0],
                self.rvals[index + 1][1] - point[1],
            ]
            r_left = [
                self.rvals[index - 1][0] - point[0],
                self.rvals[index - 1][1] - point[1],
            ]
            r_min = [self.rvals[index][0] - point[0], self.rvals[index][1] - point[1]]
            if np.dot(r_right, r_min) > 0:
                second_distance = left_point_distance
                second_t = self.dom[index - 1]
                points_distance = np.linalg.norm(
                    np.array(self.rvals[index - 1]) - np.array(closest_point)
                )
            else:
                second_distance = right_point_distance
                second_t = self.dom[index + 1]
                points_distance = np.linalg.norm(
                    np.array(self.rvals[index + 1]) - np.array(closest_point)
                )

        if np.fabs(second_distance + min_dist - points_distance) > self.tolerance:
            not_online_error = "The point {} is not on the lines segments, tolerance violated by {} times.\nThe distances are {}, {}, {}.".format(
                point,
                np.fabs(second_distance + min_dist - points_distance) / self.tolerance,
                second_distance,
                min_dist,
                points_distance,
            )
            raise ValueError(not_online_error)
        else:
            t_interpolated = (
                t * second_distance / points_distance
                + second_t * min_dist / points_distance
            )

        self.point_to_t_dict[point] = t_interpolated

        return t_interpolated

