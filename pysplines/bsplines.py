"""
TODO:
    - Currently only uniform knot vectors are implemented. Need to extend to non uniform case.
    - We should use sympy points (vectors) for sympy_Bspline.cv instead of floats
    - Should I make ALexpression a function with attributes instead of a class?
    - Can we inherit all necessary arithmetic properties of ALexpression from sympy?
"""
import numpy as np
import sympy
from sympy.functions.special.bsplines import bspline_basis as sympy_bspline_basis
import math
from scipy.integrate import simps
import warnings

from pysplines.alexpression import ALexpression
from pysplines.alexpression import is_numeric_argument


class CoreBspline:
    def __init__(self, cv, degree=3, n=100, periodic=False):
        """
        Clamped B-Spline with sympy: Core class

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
        self.kv = self.construct_knot_vector()
        self.dom = np.linspace(0, self.max_param, self.n)

        self.point_to_t_dict = {}
        self.tolerance = 1.0e-6
        self.bspline_basis = self.construct_bspline_basis()
        self.bspline = self.construct_bspline_expression()

    def construct_knot_vector(self):
        """
        Calculates knot vector of a uniform B-spline
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
        Returns:
            list of B-spline basis functions in ALexpression format
        """
        bspline_basis = [
            ALexpression(
                sympy_bspline_basis(self.degree, tuple(self.kv), i, self.x)  # 1.0 *
            )
            for i in range(len(self.cv))
        ]
        return bspline_basis

    def construct_bspline_expression(self):
        """
        Returns:
            N-dimensional list (N == self.space_dimension) of parametrized B-spline surface components
        """
        bspline_expression = [0 for s in range(self.space_dimension)]
        for i in range(len(self.cv)):
            for j in range(self.space_dimension):
                bspline_expression[j] += self.cv[i][j] * self.bspline_basis[i].aform
        return [
            ALexpression(bspline_expression[i]) for i in range(self.space_dimension)
        ]

    def evaluate_expression(self, expression, domain=None):
        """
        Given an ALexpression expression, calculates numerical values of the expression
        at the domain points.
        domain only refers to internal parameter, not the physical space points

        : param domain: single value or a list of values to evaluate the expression at

        Returns:
            List of the expression values.
            If there's only one element to return in list, return the element instead.
            (The last is the case when a point or t is given)
        """
        if is_numeric_argument(domain):
            domain = (domain,)
        elif domain is None:
            domain = self.dom

        expression_val = []
        if isinstance(expression, list):
            n = len(expression)
            for r in domain:
                val = [expression[i](r) for i in range(n)]
                expression_val.append(val)
        elif isinstance(expression, ALexpression):
            for r in domain:
                val = expression(r)
                expression_val.append(val)
        else:
            raise NotImplementedError

        if len(expression_val) == 1:
            expression_val = expression_val[0]

        return expression_val


class Bspline(CoreBspline):
    def __init__(self, cv, degree=3, n=100, periodic=False, **kwargs):
        super().__init__(cv, degree=degree, n=n, periodic=periodic)

        self._bspline_derivative = None
        self._bspline_hessian = None
        self.__arc_length = None
        self.__normal = None
        self.__curvature = None
        self.__displacement = None

        if kwargs.get("normalize_points", True):
            self.normalize_points(self.n)
        else:
            self.bspline_get_surface()

        self.is_bspline_refined = kwargs.get("refine", False)
        if self.is_bspline_refined:
            self.curvature_tolerance_angle = kwargs.get("angle_tolerance", 1.0e-2)
            self.refine_curvature()

    @property
    def bspline_derivative(self):
        if self._bspline_derivative is None:
            self._bspline_derivative = self.construct_derivative(self.bspline, 1)

        return self._bspline_derivative

    @property
    def bspline_hessian(self):
        if self._bspline_hessian is None:
            self._bspline_hessian = self.construct_derivative(self.bspline, 2)

        return self._bspline_hessian

    @property
    def _arc_length(self):
        if self.__arc_length is None:
            self.__arc_length = self.generate_arc_length()

        return self.__arc_length

    @property
    def _normal(self):
        if self.__normal is None:
            self.__normal = self.generate_normal()

        return self.__normal

    @property
    def _curvature(self):
        if self.__curvature is None:
            self.__curvature = self.generate_curvature()

        return self.__curvature

    @property
    def _displacement(self):
        if self.__displacement is None:
            self.__displacement = self.generate_displacements()

        return self.__displacement

    def get_t_from_point(self, point):
        """
        Given a point, we check if the point lies on the B-spline,
        and what internal parameter 't' it corresponds to.

        Returns:
            Interpolated value of t, such that B-spline(t) = point.

        TODO:
        - Now we only return t for points that are already on the existing lines.
          However, this is not always correct - if the bspline discretization is
          too coarse, some of the real points won't be on the existing lines.
          We need to check if the 'point' is actually near the point, predicted by 't'.
        """
        point = tuple(point)
        if point in self.point_to_t_dict:
            return self.point_to_t_dict[point]

        distance_list = list(
            map(lambda r: np.linalg.norm(np.array(r) - np.array(point)), self.rvals)
        )
        index = np.argmin(distance_list)

        t = self.dom[index]
        min_dist = distance_list[index]
        closest_point = self.rvals[index]

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
            not_on_line_error = (
                "The point {} is not on the lines segments, "
                "tolerance violated by {} times.\n"
                "The distances are {}, {}, {}."
            )
            raise ValueError(
                not_on_line_error.format(
                    point,
                    np.fabs(second_distance + min_dist - points_distance)
                    / self.tolerance,
                    second_distance,
                    min_dist,
                    points_distance,
                )
            )
        else:
            t_interpolated = (
                t * second_distance / points_distance
                + second_t * min_dist / points_distance
            )

        self.point_to_t_dict[point] = t_interpolated

        return t_interpolated

    def evaluate_expression(self, expression, point=None, pointset=None, *, t=None):
        """
        Given an ALexpression expression, calculates numerical values of the expression.
        If a point in physical space is given, converts the point to internal parameter.
        If a set of points (pointset) is given, conversts the points to internal parameter.
        If a internal parameter t is given, calculates the numerical value straight away.

        Returns:
            List of the expression values.
            If there's only one element to return in list, return the element instead.
            (The last is the case when a point or t is given)
        """

        if t is not None:
            domain = (t,)
        elif point is None and pointset is None:
            domain = self.dom
        elif point is not None:
            domain = (self.get_t_from_point(point),)
        elif pointset is not None:
            domain = (self.get_t_from_point(point) for point in pointset)

        return super().evaluate_expression(expression, domain=domain)

    def construct_derivative(self, expression, order):
        if isinstance(expression, list):
            derivative = [self.construct_derivative(exp, order) for exp in expression]
        elif isinstance(expression, ALexpression):
            derivative = ALexpression(sympy.diff(expression.aform, self.x, order))
        else:
            raise NotImplementedError
        return derivative

    def plot(self, linetype="-", window=None, **kwargs):
        if window is None:
            import matplotlib.pyplot as plt

            window = plt
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

    def plot_cv(self, window=None):
        if window is None:
            import matplotlib.pyplot as plt

            window = plt
        window.plot(
            self.cv[:, 0], self.cv[:, 1], "o", markersize=4, c="black", mfc="none"
        )

    def bspline_get_surface(self):
        """
        Evaluates the (self.space_dimension) - dimensional B-spline surface over
        the full domain, stores the radius-vector values in self.rvals
        Truncates the coordinates up to the self.tolerance level.
        """
        self.rvals = self.evaluate_expression(self.bspline)
        for i in range(len(self.rvals)):
            self.point_to_t_dict[tuple(self.rvals[i])] = self.dom[i]
            for j in range(self.space_dimension):
                self.rvals[i][j] = round(
                    self.rvals[i][j], -int(math.log10(self.tolerance))
                )
        if self.degree == 1:
            self._insert_surface_points()

    def _insert_surface_points(self):
        """
        For splines of degree 1, the spline passes through the control points.
        Naively iterate over all points and find proper places to add the control
        points.
        Mutates ``self.rvals``.
        """

        def collinear(p0, p1, p2, tolerance=1.0e-12):
            x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
            x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
            offset = abs(x1 * y2 - x2 * y1)
            return offset < tolerance

        vertices = [[_cv for _cv in cv] for cv in self.cv]
        current_vertex_index = 1

        rvals = [vertices[0]]
        for rval_index, rval in enumerate(self.rvals[1:-1], start=1):
            if not collinear(
                vertices[current_vertex_index - 1], vertices[current_vertex_index], rval
            ):
                rvals.append(vertices[current_vertex_index])
                current_vertex_index += 1
            rvals.append(rval)

        rvals.append(vertices[-1])
        self.rvals = rvals

    def normalize_points(self, n):
        """
        Reconstructs the distribution of the internal parameter t in self.dom which corresponds to
        a uniform distribution of the physical points.

        When the internal parameter t of a B-spline is uniformly distributed (from 0 to 1),
        it doesn't presume the physical points are distributed uniformly along the surface.
        First, we generate a very fine discretization of the B-spline we have with 3000 points,
        and then remove the points until they are distributed uniformly, and their number (almost)
        equals to the user specified number.

        TODO:
            - Find a way to estimate the minimum necessary number of points on the surface,
                when we construct the fine discretization.
        """
        self.n = 3000
        self.dom = np.linspace(0, self.max_param, self.n)
        self.bspline_get_surface()
        self.n = n

        L = self.arc_length()

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
        self.bspline_get_surface()
        self.n = len(self.dom)

    def dots_angles(self, direction="forward"):
        """
        Iterates through the self.rvals points in the given direction and calculates
        the angle between two vectors, v1 and v2:
        v1: (current point, next point)
        v2: (current point, after next point)

        Returns:
            List of angles
        """
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
        """
        Some of the B-spline parts can be underresolved, which yields sharp corners
        in the high-curvature regions.
        We refine the surface such that the three consecutive points (1,2,3) lie
        almost on the same line, such that the angle((2,1), (2,3)) <= self.curvature_tolerance_angle.
        We iterate and refine the surface both in the forward and backwards directions.

        This method will increase the number of points on the surface.
        """

        tolerance_angle = self.curvature_tolerance_angle

        proper_t_dist = []
        proper_t_dist.append(self.dom[0])
        angles = self.dots_angles()
        tresh_angle = tolerance_angle
        for i in range(len(self.dom) - 2):
            if abs(angles[i]) < abs(tresh_angle):
                proper_t_dist.append(self.dom[i + 1])
            else:
                n_insert_points = int(angles[i] / tresh_angle)
                dt = (self.dom[i + 2] - self.dom[i + 1]) / n_insert_points
                for j in range(n_insert_points):
                    proper_t_dist.append(self.dom[i + 1] + j * dt)
        proper_t_dist.append(self.dom[-1])
        self.dom = proper_t_dist
        self.bspline_get_surface()
        self.n = len(self.dom)

        proper_t_dist = []
        proper_t_dist.append(self.dom[-1])
        angles = self.dots_angles(direction="backward")[::-1]
        tresh_angle = tolerance_angle
        for i in range(len(self.dom) - 2):
            k = -1 - i
            if abs(angles[k]) < abs(tresh_angle):
                proper_t_dist.append(self.dom[k - 1])
            else:
                n_insert_points = int(angles[k] / tresh_angle)
                dt = (self.dom[k - 2] - self.dom[k - 1]) / n_insert_points
                for j in range(n_insert_points):
                    proper_t_dist.append(self.dom[k - 1] + j * dt)
        proper_t_dist.append(self.dom[0])
        self.dom = proper_t_dist[::-1]
        self.bspline_get_surface()
        self.n = len(self.dom)

    def generate_arc_length(self):
        L = ALexpression(
            sympy.sqrt(np.inner(self.bspline_derivative, self.bspline_derivative).aform)
        )
        return L

    def generate_normal(self):
        if self.space_dimension > 2:
            warnings.warn(
                "Only 2D version of {} is implemented. Current space dimension is {}".format(
                    "normal", self.space_dimension
                )
            )
        return [
            -self.bspline_derivative[1] / self._arc_length,
            self.bspline_derivative[0] / self._arc_length,
        ]

    def generate_curvature(self):
        if self.space_dimension > 2:
            warnings.warn(
                "Only 2D version of {} is implemented. Current space dimension is {}".format(
                    "normal", self.space_dimension
                )
            )
        curvature = (
            self.bspline_derivative[0].aform * self.bspline_hessian[1].aform
            - self.bspline_derivative[1].aform * self.bspline_hessian[0].aform
        ) / self._arc_length.aform ** 3.0
        return ALexpression(curvature)

    @property
    def surface_area(self):
        """
        Returns the area under the B-spline curve.
        """
        y = np.array(self.rvals)[:, 1]
        x_t = self.evaluate_expression(self.bspline_derivative[0])
        surface_area = simps(y * x_t, self.dom)
        return surface_area

    def mass_matrix(self, DLMM=False):
        """
        Mass matrix M measures how much the B-spline basis functions overlap.
        M_{ij} = int dl(t) B_i(t) B_j(t)

        We can reduce the mass matrix to a Diagonally Lumped Mass Matrix (DLMM).
        Then only the diagonal elements appear.

        We use the mass matrix to correct the gradients with respect to control
        points positions, i.e. map the discrete gradient vector to the parameter-independent
        case. Read about it here: http://dx.doi.org/10.1016/j.cad.2016.06.002

        Returns:
            Mass Matrix
        """

        L = self.arc_length()

        M = np.zeros((len(self.cv), len(self.cv)))
        bspline_basis_eval = np.array(self.evaluate_expression(self.bspline_basis))

        for i in range(len(self.cv)):
            for j in range(len(self.cv)):
                if i > j:
                    M[i, j] = M[j, i]
                else:
                    if (DLMM and i == j) or not DLMM:
                        M[i, j] = simps(
                            bspline_basis_eval[:, i] * bspline_basis_eval[:, j] * L,
                            self.dom,
                        )

        return M

    def generate_displacements(self):
        """
        Generates ALexpression for B-spline displacement with respect to control points
        """
        displacements = []
        for control_point_number in range(len(self.cv)):
            cp_displacement = [
                self.bspline_basis[control_point_number]
                for i in range(self.space_dimension)
            ]
            displacements.append(cp_displacement)
        return displacements

    def arc_length(self, point=None):
        return self.evaluate_expression(self._arc_length, point=point)

    def normal(self, point=None):
        return self.evaluate_expression(self._normal, point=point)

    def curvature(self, point=None):
        return self.evaluate_expression(self._curvature, point=point)

    def displacement(self, point=None, control_point_number=None):
        if control_point_number is not None:
            return self.evaluate_expression(
                self._displacement[control_point_number], point=point
            )

        return self.evaluate_expression(self._displacement, point=point)
