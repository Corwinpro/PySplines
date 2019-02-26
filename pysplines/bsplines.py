"""
TODO:
    - We should use sympy points (vectors) for sympy_Bspline.cv instead of floats
    - Should I make ALexpression a function with attributes instead of a class?
"""
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
    - lform: fully lambdified form of the expression
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

    def __mul__(self, other):
        if isinstance(other, ALexpression):
            return ALexpression(self.aform * other.aform)
        else:
            raise ValueError("int, float value or Parameter is required")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, ALexpression):
            return ALexpression(self.aform + other.aform)
        else:
            raise ValueError("int, float value or Parameter is required")

    def __radd__(self, other):
        return self.__add__(other)


class CoreBspline:
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
        """
         I multiply by 1.0 here for the following reason:
         - without 1.0 * the bspline_basis[i].aform is type Piecewise
         - with 1.0 * it is type <class 'sympy.core.mul.Mul'>
         When I lambdify and evaluate expression it becomes
         - np.ndarray() without multiplication
         - np.float() with multiplication (which I need)
         Is it a bug?
        """
        bspline_basis = [
            ALexpression(1.0 * sympy.bspline_basis(self.degree, self.kv, i, self.x))
            for i in range(len(self.cv))
        ]
        return bspline_basis

    def construct_bspline_expression(self):
        bspline_expression = [0] * self.space_dimension

        for i in range(len(self.cv)):
            for j in range(self.space_dimension):
                bspline_expression[j] += self.cv[i][j] * self.bspline_basis[i].aform
        return [
            ALexpression(bspline_expression[i]) for i in range(self.space_dimension)
        ]

    def bspline_getSurface(self, domain=None):

        self.rvals = self.evaluate_expression(self.bspline, domain=domain)

        for i in range(len(self.rvals)):
            self.point_to_t_dict[tuple(self.rvals[i])] = self.dom[i]
            for j in range(self.space_dimension):
                self.rvals[i][j] = (
                    math.trunc(self.rvals[i][j] / self.tolerance) * self.tolerance
                )

    def evaluate_expression(self, expression, point=None, domain=None):
        """
        Given a sympy expression, a point (or set of points), calculates
        values of the expression at the point(s).

        Returns: a list of the expression values

        TODO: check all possible usage cases
        """
        if point is None and domain is None:
            domain = self.dom
        elif point is not None:
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
                self.rvals[index + 1][i] - point[i] for i in range(self.space_dimension)
            ]
            r_left = [
                self.rvals[index - 1][i] - point[i] for i in range(self.space_dimension)
            ]
            r_min = [
                self.rvals[index][i] - point[i] for i in range(self.space_dimension)
            ]

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
            not_on_line_error = "The point {} is not on the lines segments, tolerance violated by {} times.\nThe distances are {}, {}, {}.".format(
                point,
                np.fabs(second_distance + min_dist - points_distance) / self.tolerance,
                second_distance,
                min_dist,
                points_distance,
            )
            raise ValueError(not_on_line_error)
        else:
            t_interpolated = (
                t * second_distance / points_distance
                + second_t * min_dist / points_distance
            )

        self.point_to_t_dict[point] = t_interpolated

        return t_interpolated

    def get_displacement_from_point(self, point, controlPointNumber):
        t = self.get_t_from_point(point)
        displacement = self.bspline_basis[controlPointNumber][t]
        return [displacement for i in range(self.space_dimension)]

    def expression_from_point(self, expression, point):
        raise NotImplementedError


class Bspline(CoreBspline):
    def __init__(self, cv, degree=3, n=100, periodic=False, **kwargs):
        super().__init__(cv, degree=degree, n=n, periodic=periodic)

        self.bspline_derivative = self.construct_derivative(self.bspline, 1)

        self.normalize_points(self.n)

        self.is_bspline_refined = kwargs.get("refine", False)
        if self.is_bspline_refined:
            self.curvature_tolerance_angle = kwargs.get("angle_tolerance", 1.0e-2)
            self.refine_curvature()

    def construct_derivative(self, expression, order):

        if isinstance(expression, list):
            derivative = [self.construct_derivative(exp, order) for exp in expression]
        elif isinstance(expression, ALexpression):
            derivative = ALexpression(sympy.diff(expression.aform, self.x, order))
        else:
            raise NotImplementedError
        return derivative

    def plot(self, linetype="-", window=plt, **kwargs):

        window.plot(
            np.array(self.rvals)[:, 0],
            np.array(self.rvals)[:, 1],
            linetype,
            color=kwargs.get("color", "black"),
        )
        self.plot_cv(window)
        show = kwargs.get("show", True)
        if show:
            window.show()

    def plot_cv(self, window=plt):

        window.plot(
            self.cv[:, 0], self.cv[:, 1], "o", markersize=4, c="black", mfc="none"
        )

    def normalize_points(self, n):

        self.n = 3000
        self.dom = np.linspace(0, self.max_param, self.n)
        self.bspline_getSurface()
        self.n = n

        L = ALexpression(
            sympy.sqrt(np.inner(self.bspline_derivative, self.bspline_derivative).aform)
        )
        L = self.evaluate_expression(L)

        totalLength = simps(L, self.dom)
        avgDistance = totalLength / n

        _tmp_dist = 0.0
        proper_t_dist = []
        proper_t_dist.append(self.dom[0])

        for i in range(1, len(self.dom) - 1):
            if _tmp_dist < avgDistance:
                _tmp_dist += (
                    sum(
                        (self.rvals[i][j] - self.rvals[i - 1][j]) ** 2.0
                        for j in range(self.space_dimension)
                    )
                ) ** 0.5
            else:
                _tmp_dist = 0
                proper_t_dist.append(self.dom[i])
        proper_t_dist.append(self.dom[-1])

        self.dom = proper_t_dist
        self.bspline_getSurface()
        self.n = len(self.dom)

    def dots_angles(self, direction="forward"):
        if direction == "backward":
            # Move backwards along the surface pointlist
            radius_vector = [
                np.array(self.rvals)[::-1][:, i] for i in range(self.space_dimension)
            ]
        else:
            # Else move as usual
            radius_vector = [
                np.array(self.rvals)[:, i] for i in range(self.space_dimension)
            ]

        angles = []
        for i in range(len(self.rvals) - 2):
            v1_u = [x[i + 1] - x[i] for x in radius_vector]
            v1_u /= np.linalg.norm(v1_u)
            v2_u = [x[i + 2] - x[i] for x in radius_vector]
            v2_u /= np.linalg.norm(v2_u)
            angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
            angles.append(angle)

        return angles

    def refine_curvature(self):

        tolerance_angle = self.curvature_tolerance_angle

        proper_t_dist = []
        proper_t_dist.append(self.dom[0])
        angles = self.dots_angles()
        tresh_angle = tolerance_angle
        for i in range(len(self.dom) - 2):
            if angles[i] ** 2.0 < tresh_angle ** 2.0:
                proper_t_dist.append(self.dom[i + 1])
            else:
                n_insert_points = int(angles[i] / tresh_angle)
                dt = (self.dom[i + 2] - self.dom[i + 1]) / n_insert_points
                for j in range(n_insert_points):
                    proper_t_dist.append(self.dom[i + 1] + j * dt)
        proper_t_dist.append(self.dom[-1])
        self.dom = proper_t_dist
        self.bspline_getSurface()
        self.n = len(self.dom)

        proper_t_dist = []
        proper_t_dist.append(self.dom[-1])
        angles = self.dots_angles(direction="backward")[::-1]
        tresh_angle = tolerance_angle
        for i in range(len(self.dom) - 2):
            k = -1 - i
            if angles[k] ** 2.0 < tresh_angle ** 2.0:
                proper_t_dist.append(self.dom[k - 1])
            else:
                n_insert_points = int(angles[k] / tresh_angle)
                dt = (self.dom[k - 2] - self.dom[k - 1]) / n_insert_points
                for j in range(n_insert_points):
                    proper_t_dist.append(self.dom[k - 1] + j * dt)
        proper_t_dist.append(self.dom[0])
        self.dom = proper_t_dist[::-1]
        self.bspline_getSurface()
        self.n = len(self.dom)

    def normal(self):
        raise NotImplementedError

    def curvature(self):
        raise NotImplementedError

    def surface_area(self):
        raise NotImplementedError

    def mass_matrix(self):
        raise NotImplementedError

    def displacement(self):
        raise NotImplementedError
