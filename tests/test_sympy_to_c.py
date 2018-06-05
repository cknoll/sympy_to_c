#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sympy_to_c` package."""


import unittest
import sympy as sp
import numpy as np
import os

from ipydex import IPS

from sympy_to_c import sympy_to_c as sp2c


# noinspection PyPep8Naming, PyTypeChecker
class TestSympy_to_c(unittest.TestCase):
    """Tests for `sympy_to_c` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        x1, x2, x3 = self.xx = sp.symbols("x1, x2, x3")

        self.e1 = x1*x2 + x3

        self.M1 = sp.Matrix([self.e1, sp.sin(x1)*sp.exp(x3)])

        np.random.seed(1840)
        N = 100
        self.XX = np.random.random((N, len(self.xx)))

    def tearDown(self):
        """Tear down test fixtures, if any."""
        while sp2c.created_so_files:
            sofilepath = sp2c.created_so_files.pop()
            os.remove(sofilepath)

    def test_scalar_expression(self):
        """Test something."""

        e1_c_func = sp2c.convert_to_c(self.xx, self.e1, cfilepath="scalar.c",
                                      use_exisiting_so=False)
        e1_l_func = sp.lambdify(self.xx, self.e1)

        for xx in self.XX:
            self.assertAlmostEqual(e1_c_func(*xx), e1_l_func(*xx))

    def test_matrix_expression(self):
        """Test something."""

        M1_c_func = sp2c.convert_to_c(self.xx, self.M1, cfilepath="matrix.c",
                                      use_exisiting_so=False)
        M1_l_func = sp.lambdify(self.xx, self.M1)

        for xx in self.XX:
            self.assertTrue(np.allclose(M1_c_func(*xx), M1_l_func(*xx)))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
