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
        self.aform = sympy.simplify(sympy_expression)
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

    def __truediv__(self, other):
        if isinstance(other, ALexpression):
            return ALexpression(self.aform / other.aform)
        elif is_numeric_argument(other):
            return ALexpression(self.aform / other)
        else:
            raise ValueError(
                "int or float value is required. Current is {}".format(type(other))
            )

    def __rtruediv__(self, other):
        if isinstance(other, ALexpression):
            return ALexpression(other.aform / self.aform)
        elif is_numeric_argument(other):
            return ALexpression(other / self.aform)
        else:
            raise ValueError(
                "int or float value is required. Current is {}".format(type(other))
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
            return ValueError("Only ALexpression or Sympy expressions or numerical arguments can be compared")
