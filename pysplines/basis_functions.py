"""
This is a fork from sympy implementation of bspline basis functions,
when we fixed _add_splines summation functionality
"""

from __future__ import print_function, division

from sympy.core import S, sympify
from sympy.core.compatibility import range
from sympy.functions import Piecewise, piecewise_fold
from sympy.sets.sets import Interval

from functools import lru_cache


def _add_splines(c, b1, d, b2):
    """Construct c*b1 + d*b2."""
    if b1 == S.Zero or c == S.Zero:
        rv = piecewise_fold(d * b2)
    elif b2 == S.Zero or d == S.Zero:
        rv = piecewise_fold(c * b1)
    else:
        new_args = []
        n_intervals = len(b1.args)
        # Just combining the Piecewise without any fancy optimization
        p1 = piecewise_fold(c * b1)
        p2 = piecewise_fold(d * b2)

        # Search all Piecewise arguments except (0, True)
        p2args = list(p2.args[:-1])

        # This merging algorithm assumes the conditions in
        # p1 and p2 are sorted
        for arg in p1.args[:-1]:
            # Conditional of Piecewise are And objects
            # the args of the And object is a tuple of two
            # Relational objects the numerical value is in the .rhs
            # of the Relational object
            expr = arg.expr
            cond = arg.cond

            lower = cond.args[0].rhs

            # Check p2 for matching conditions that can be merged
            for i, arg2 in enumerate(p2args):
                expr2 = arg2.expr
                cond2 = arg2.cond

                lower_2 = cond2.args[0].rhs
                upper_2 = cond2.args[1].rhs

                if cond2 == cond:
                    # Conditions match, join expressions
                    expr += expr2
                    # Remove matching element
                    del p2args[i]
                    # No need to check the rest
                    break
                elif lower_2 < lower and upper_2 <= lower:
                    # Check if arg2 condition smaller than arg1,
                    # add to new_args by itself (no match expected
                    # in p1)
                    new_args.append(arg2)
                    del p2args[i]
                    break

            # Checked all, add expr and cond
            new_args.append((expr, cond))

        # Add remaining items from p2args
        new_args.extend(p2args)

        # Add final (0, True)
        new_args.append((0, True))

        rv = Piecewise(*new_args)

    return rv.expand()


@lru_cache()
def bspline_basis(d, knots, n, x):
    """The `n`-th B-spline at `x` of degree `d` with knots.

    B-Splines are piecewise polynomials of degree `d` [1]_.  They are
    defined on a set of knots, which is a sequence of integers or
    floats.

    The 0th degree splines have a value of one on a single interval:

        >>> from sympy import bspline_basis
        >>> from sympy.abc import x
        >>> d = 0
        >>> knots = range(5)
        >>> bspline_basis(d, knots, 0, x)
        Piecewise((1, (x >= 0) & (x <= 1)), (0, True))

    For a given ``(d, knots)`` there are ``len(knots)-d-1`` B-splines
    defined, that are indexed by ``n`` (starting at 0).

    Here is an example of a cubic B-spline:

        >>> bspline_basis(3, range(5), 0, x)
        Piecewise((x**3/6, (x >= 0) & (x <= 1)),
                  (-x**3/2 + 2*x**2 - 2*x + 2/3,
                  (x >= 1) & (x <= 2)),
                  (x**3/2 - 4*x**2 + 10*x - 22/3,
                  (x >= 2) & (x <= 3)),
                  (-x**3/6 + 2*x**2 - 8*x + 32/3,
                  (x >= 3) & (x <= 4)),
                  (0, True))

    By repeating knot points, you can introduce discontinuities in the
    B-splines and their derivatives:

        >>> d = 1
        >>> knots = [0, 0, 2, 3, 4]
        >>> bspline_basis(d, knots, 0, x)
        Piecewise((-x/2 + 1, (x >= 0) & (x <= 2)), (0, True))

    It is quite time consuming to construct and evaluate B-splines. If
    you need to evaluate a B-splines many times, it is best to
    lambdify them first:

        >>> from sympy import lambdify
        >>> d = 3
        >>> knots = range(10)
        >>> b0 = bspline_basis(d, knots, 0, x)
        >>> f = lambdify(x, b0)
        >>> y = f(0.5)

    See Also
    ========

    bsplines_basis_set

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/B-spline

    """
    knots = tuple([sympify(k) for k in knots])
    d = int(d)
    n = int(n)
    n_knots = len(knots)
    n_intervals = n_knots - 1
    if n + d + 1 > n_intervals:
        raise ValueError("n + d + 1 must not exceed len(knots) - 1")
    if d == 0:
        result = Piecewise(
            (S.One, Interval(knots[n], knots[n + 1]).contains(x)), (0, True)
        )
    elif d > 0:
        denom = knots[n + d + 1] - knots[n + 1]
        if denom != S.Zero:
            B = (knots[n + d + 1] - x) / denom
            b2 = bspline_basis(d - 1, knots, n + 1, x)
        else:
            b2 = B = S.Zero

        denom = knots[n + d] - knots[n]
        if denom != S.Zero:
            A = (x - knots[n]) / denom
            b1 = bspline_basis(d - 1, knots, n, x)
        else:
            b1 = A = S.Zero

        result = _add_splines(A, b1, B, b2)
    else:
        raise ValueError("degree must be non-negative: %r" % n)
    return result
