import unittest
import sympy

from pysplines.alexpression import ALexpression


class TestALexpression(unittest.TestCase):
    def test_lform(self):
        x = sympy.var("x")
        expression = x ** 2 + 3 * x + 4
        al_expression = ALexpression(expression)
        self.assertEqual(al_expression[2.0], 14.0)

    def test_plus(self):
        x = sympy.var("x")
        expression_1 = x ** 2 + 3 * x - 4
        expression_2 = x - 1
        expression_3 = x ** 2 + 4 * x - 5

        al_expression_1 = ALexpression(expression_1)
        al_expression_2 = ALexpression(expression_2)
        al_expression_3 = ALexpression(expression_3)

        self.assertEqual(al_expression_1 + al_expression_2, x ** 2 + 4 * x - 5)
        self.assertEqual(al_expression_1 + al_expression_2, al_expression_3)

        self.assertEqual((al_expression_1 + al_expression_2)[1], 0)

    def test_negative(self):
        x = sympy.var("x")
        expression_1 = x ** 2 + 3 * x - 4
        al_expression_1 = ALexpression(expression_1)

        self.assertEqual((-al_expression_1)[1], 0)
        self.assertEqual(-al_expression_1 + (x ** 2 + 3 * x - 4), 0)

    def test_divmult(self):
        x = sympy.var("x")
        expression_1 = x ** 2 + 3 * x - 4
        expression_2 = x - 1
        expression_3 = x + 4

        al_expression_1 = ALexpression(expression_1)
        al_expression_2 = ALexpression(expression_2)
        al_expression_3 = ALexpression(expression_3)

        self.assertEqual(al_expression_1 / al_expression_2 / al_expression_3, 1)
        self.assertEqual(al_expression_3 * al_expression_2 / al_expression_1, 1)

        self.assertEqual((al_expression_1 / al_expression_2)[1], 5)
        self.assertEqual((al_expression_3 * al_expression_2)[2], 6)


if __name__ == "__main__":
    unittest.main()
