import sympy

from pysplines.alexpression import ALexpression


def test_lform():
    x = sympy.var("x")
    expression = x ** 2 + 3 * x + 4
    al_expression = ALexpression(expression)
    assert al_expression(2.0) == 14.0


def test_plus():
    x = sympy.var("x")
    expression_1 = x ** 2 + 3 * x - 4
    expression_2 = x - 1
    expression_3 = x ** 2 + 4 * x - 5

    al_expression_1 = ALexpression(expression_1)
    al_expression_2 = ALexpression(expression_2)
    al_expression_3 = ALexpression(expression_3)

    assert al_expression_1 + al_expression_2 == x ** 2 + 4 * x - 5
    assert al_expression_1 + al_expression_2 == al_expression_3
    assert (al_expression_1 + al_expression_2)(1) == 0


def test_negative():
    x = sympy.var("x")
    expression_1 = x ** 2 + 3 * x - 4
    al_expression_1 = ALexpression(expression_1)

    assert (-al_expression_1)(1) == 0
    assert -al_expression_1 + (x ** 2 + 3 * x - 4) == 0


def test_divmult():
    x = sympy.var("x")
    expression_1 = x ** 2 + 3 * x - 4
    expression_2 = x - 1
    expression_3 = x + 4

    al_expression_1 = ALexpression(expression_1)
    al_expression_2 = ALexpression(expression_2)
    al_expression_3 = ALexpression(expression_3)

    trivial_expression_1 = al_expression_1 / al_expression_2 / al_expression_3
    trivial_expression_2 = al_expression_3 * al_expression_2 / al_expression_1
    trivial_expression_1.simplify()
    trivial_expression_2.simplify()

    assert trivial_expression_1 == 1
    assert trivial_expression_2 == 1

    assert (al_expression_1 / al_expression_2)(1) == 5
    assert (al_expression_3 * al_expression_2)(2) == 6
