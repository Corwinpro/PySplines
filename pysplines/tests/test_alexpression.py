import unittest

import sympy

from pysplines.alexpression import ALexpression


class TestALexpression(unittest.TestCase):
    def setUp(self):
        self.x = sympy.var("x")
        self.expression = self.x ** 2.0 + 3.0 * self.x + 4.0

    def test_init(self):
        al_expression = ALexpression(self.expression)
        self.assertIs(al_expression.aform, self.expression)
        self.assertIs(al_expression._ALexpression__initial_aform, self.expression)
        self.assertEqual(al_expression.t, (self.x,))
        self.assertIsNone(al_expression.lform)

    def test___call__(self):
        al_expression = ALexpression(self.expression)
        self.assertEqual(al_expression(2.0), 14.0)

        self.assertEqual(al_expression(2), 14.0)

    def test___call__raises(self):
        al_expression = ALexpression(self.expression)
        with self.assertRaises(TypeError):
            al_expression("abc")

    def test_no_free_variables(self):
        al_expression = ALexpression(sympy.pi)
        self.assertAlmostEqual(al_expression(2), 3.1415926, places=6)

    def test_multiply_number(self):
        # Given
        al_expression = ALexpression(self.expression)

        # When
        new_expression = al_expression * 42

        # Then
        self.assertIsInstance(new_expression, ALexpression)
        self.assertEqual(
            sympy.simplify(
                new_expression.aform
                - (42.0 * self.x ** 2.0 + 42.0 * 3.0 * self.x + 42.0 * 4.0)
            ),
            0.0,
        )

    def test_multiply_with_alexpression(self):
        # Given
        al_expression_1 = ALexpression(self.expression)
        al_expression_2 = ALexpression(self.x)

        # When
        new_expression = al_expression_1 * al_expression_2

        # Then
        self.assertEqual(
            sympy.simplify(
                new_expression.aform - (self.x ** 3.0 + 3 * self.x ** 2 + 4.0 * self.x)
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
