import warnings
import sympy


def is_numeric_argument(arg):
    _is_numeric = isinstance(arg, (int, float))
    return _is_numeric


class ALexpression:
    """
    A dual Analytical-Lambdified form expression of a single variable

    Attributes:
    - aform: analytical sympy form of the expression
    - lform: fully lambdified form of the expression
    """

    def __init__(self, sympy_expression):
        self.aform = sympy_expression
        self.__initial_aform = sympy_expression
        self.t = tuple(sympy_expression.free_symbols)
        if len(self.t) > 1:
            warnings.warn(
                "Be careful with the lambdified expression {}, "
                "as it has more than one free symbol.".format(self.aform)
            )

        self.lform = None

    def __repr__(self):
        is_lambdified = self.lform is not None
        return "ALexpression({}, lambdified: {})".format(self.aform, is_lambdified)

    def __str__(self):
        return self.__repr__()

    def __call__(self, t):
        if not is_numeric_argument(t):
            raise TypeError("int or float value is required")

        if self.lform is None:
            self.aform = self.simplify()
            self.lform = sympy.lambdify(self.t, self.aform)

        try:
            value = self.lform(t)
        except ValueError:
            self.aform = self.simplify(level=1)
            self.lform = sympy.lambdify(self.t, self.aform)
            value = self.lform(t)
        except TypeError:
            # Deal with the case when the lambdified expression
            # has no free variables (e.g., is a constant). In
            # this case, try to evaluate the expression with no
            # input arguments
            value = self.lform()
        return float(value)

    def __mul__(self, other):
        if isinstance(other, ALexpression):
            return ALexpression(self.aform * other.aform)
        elif isinstance(other, sympy.Expr) or is_numeric_argument(other):
            return ALexpression(self.aform * other)
        else:
            raise TypeError("int, float value or ALexpression is required")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, ALexpression):
            return ALexpression(self.aform + other.aform)
        elif isinstance(other, sympy.Expr):
            return ALexpression(self.aform + other)
        else:
            raise TypeError("int, float value or ALexpression is required")

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        if isinstance(other, ALexpression):
            return ALexpression(self.aform / other.aform)
        elif is_numeric_argument(other):
            return ALexpression(self.aform / other)
        elif isinstance(other, sympy.Expr):
            return ALexpression(self.aform / other)
        else:
            raise TypeError(
                "int, float value or ALexpression is required. Current is {}".format(
                    type(other)
                )
            )

    def __rtruediv__(self, other):
        if isinstance(other, ALexpression):
            return ALexpression(other.aform / self.aform)
        elif is_numeric_argument(other):
            return ALexpression(other / self.aform)
        elif isinstance(other, sympy.Expr):
            return ALexpression(other / self.aform)
        else:
            raise TypeError(
                "int or float value or ALexpression is required. Current is {}".format(
                    type(other)
                )
            )

    def __neg__(self):
        return ALexpression(-self.aform)

    def __eq__(self, value):
        if isinstance(value, ALexpression):
            return self.aform == value.aform
        elif isinstance(value, sympy.Expr):
            return self.aform == sympy.simplify(value)
        elif is_numeric_argument(value):
            return self.aform == value
        else:
            return TypeError(
                "Only ALexpression or Sympy expressions or numerical arguments can be compared"
            )

    def simplify(self, level=0):
        """
        Simplify the ``self.__initial_aform``, and return the simplified
        version

        : param level: simplification level
            level == 0: basic simplification through sympy.factor. It performs
            simple transformation that puts the expression into the standard
            form p/q, which is much faster than a generic .simplify
            level == 1: full simplification through sympy.simplify
        """
        if level == 0:
            return sympy.factor(self.__initial_aform)
        elif level == 1:
            return sympy.simplify(self.__initial_aform)
