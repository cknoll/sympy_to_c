#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sympy_to_c` package."""

import unittest
import sympy as sp
from sympy.utilities.codegen import codegen
from sympy.utilities.lambdify import implemented_function
from sympy import Piecewise, Abs, ITE

import numpy as np
import os
import sys
import hashlib
import pickle

try:
    # this is handy for debugging but otherwise not needed
    from ipydex import IPS, activate_ips_on_exception
    activate_ips_on_exception()

except ImportError:
    pass

import sympy_to_c as sp2c

if sys.version_info[0] == 2:
    input = raw_input


# noinspection PyPep8Naming, PyTypeChecker
class TestSympy_to_c(unittest.TestCase):
    """Tests for `sympy_to_c` package."""

    __name__ = "test"

    def setUp(self):
        """Set up test fixtures, if any."""
        x1, x2, x3 = self.xx = sp.symbols("x1, x2, x3", complex=False, finite=True)

        self.e1 = x1*x2 + x3

        self.M1 = sp.Matrix([self.e1, sp.sin(x1)*sp.exp(x3), 42])

        np.random.seed(1840)
        N = 100
        self.XX = np.random.random((N, len(self.xx)))

        dir_of_this_file = os.path.abspath(os.path.dirname(__file__))
        self.matrix_c_file_path = os.path.join(dir_of_this_file, "matrix.c")


    def tearDown(self):
        """Tear down test fixtures, if any."""

        sp2c.unload_all_libs()

        while sp2c.created_so_files:
            so_file_path = sp2c.created_so_files.pop()
            print("deleting", so_file_path)
            os.remove(so_file_path)

    def test_01__scalar_expression(self):
        """Test conversion of simple scalar expression."""

        e1_c_func = sp2c.convert_to_c(self.xx, self.e1, c_file_path="scalar.c",
                                      use_existing_so=False)
        e1_l_func = sp.lambdify(self.xx, self.e1)

        for xx in self.XX:
            self.assertAlmostEqual(e1_c_func(*xx), e1_l_func(*xx))

    def test_02__matrix_expression(self):
        """Test conversion of simple matrix."""

        M1_c_func = sp2c.convert_to_c(self.xx, self.M1, c_file_path="matrix.c",
                                      use_existing_so=False)
        M1_l_func = sp.lambdify(self.xx, self.M1)

        for xx in self.XX:
            res1 = M1_c_func(*xx)
            res2 = M1_l_func(*xx)
            self.assertTrue(np.allclose(res1, res2))

    def test_02b__list_of_expressions(self):
        """Test conversion of simple matrix."""

        list_of_exprs = list(self.M1)
        M1_c_func = sp2c.convert_to_c(self.xx, list_of_exprs, c_file_path="matrix.c",
                                      use_existing_so=False)
        M1_l_func = sp.lambdify(self.xx, list_of_exprs)

        for xx in self.XX:
            res1 = M1_c_func(*xx)
            res2 = M1_l_func(*xx)
            self.assertTrue(np.allclose(res1, res2))

    def test_03__meta_data(self):
        """
        Background:
        convert_to_c can store almost arbitrary data inside the shared library in form of a base64 encoded dict.
        """

        # additional metadata
        amd = dict(fnordskol=23.42)
        M1_c_func = sp2c.convert_to_c(self.xx, self.M1, c_file_path="matrix.c",
                                      use_existing_so=False, additional_metadata=amd)

        # get metadata directly
        md = M1_c_func.metadata

        # load metadata from library (e.g. from a different program)
        md2 = sp2c.get_meta_data(self.matrix_c_file_path)

        # the dicts must be equal but not be identical
        assert md == md2
        assert md is not md2

        self.assertTrue(isinstance(md, dict))
        self.assertTrue("fingerprint" in md)
        self.assertTrue("timestamp" in md)
        self.assertTrue("nargs" in md)
        self.assertTrue("args" in md)

        self.assertEqual(md["nargs"], len(self.xx))
        self.assertEqual(md["fnordskol"], 23.42)
        self.assertEqual(md["args"], self.xx)

    @unittest.skip
    @unittest.expectedFailure
    def test_04__hashing1(self):

        # this test is related to https://github.com/sympy/sympy/issues/14808

        # e1 = self.M1[0]
        e1 = sp.symbols("x1")

        h1 = hashlib.sha256(pickle.dumps(e1)).hexdigest()
        M1_1 = self.M1.copy()
        h2 = hashlib.sha256(pickle.dumps(e1)).hexdigest()

        # this call changes something in the pickle representation
        codegen(("M1_00", e1), "C", "test", header=False, empty=False, argument_sequence=self.xx)

        h3 = hashlib.sha256(pickle.dumps(e1)).hexdigest()

        self.assertEqual(h1, h2)
        self.assertEqual(h1, h3)

    def test_04b__hashing3(self):

        h1 = hashlib.sha256(pickle.dumps(self.xx)).hexdigest()
        s1 = pickle.dumps(self.xx[0])

        res = codegen(("M1_00", self.xx[0]*0), "C", "M1_00", argument_sequence=self.xx)
        s2 = pickle.dumps(self.xx[0])

        h2 = hashlib.sha256(pickle.dumps(self.xx)).hexdigest()

        # this fails if we did not set special assumptions during symbol creation
        # see https://github.com/sympy/sympy/issues/14808

        self.assertEqual(h1, h2)

    def test_05__reproducible_fast_hash(self):
        """

        :return:
        """

        # TODO: this test should incorporate multiple runs from the python-interpreter
        # to ensure reproducibility

        h1 = sp2c.reproducible_fast_hash([self.M1, self.e1])
        h2 = sp2c.reproducible_fast_hash([self.M1, self.e1])

        self.assertEqual(h1, h2)
        print("This should be the same in every run: {}".format(h1))

    def test_06__use_existing(self):

        # create new so-file
        sp2c.CLEANUP = False

        M1_c_func = sp2c.convert_to_c(self.xx, self.M1, c_file_path="matrix.c",
                                      use_existing_so=False)

        # other expression but no new c-Code
        print("\n", "other expression but no new c-Code")
        M2_c_func = sp2c.convert_to_c(self.xx, self.M1*0, c_file_path="matrix.c",
                                      use_existing_so=True)

        self.assertTrue(M2_c_func.reused_c_code)

        # test that result is the same
        args = (1702, 123, -12.34)
        r1 = M1_c_func(*args).flatten()
        r2 = M2_c_func(*args).flatten()
        for i in range(len(self.M1)):
            self.assertEqual(r1[i], r2[i])

        ts2 = sp2c.get_meta_data(self.matrix_c_file_path)["timestamp"]

        # same expression -> no new code
        print("\n", " same expression -> no new code")
        M3_c_func = sp2c.convert_to_c(self.xx, self.M1, c_file_path="matrix.c",
                                      use_existing_so="smart")

        ts3 = sp2c.get_meta_data(self.matrix_c_file_path)["timestamp"]

        md3 = sp2c.get_meta_data(self.matrix_c_file_path)

        # this should work
        self.assertTrue(M3_c_func.reused_c_code)

        self.assertEqual(ts2, ts3)

        print("\n", " different expression -> new code -> new_load")
        M4_c_func = sp2c.convert_to_c(self.xx, self.M1*0, c_file_path="matrix.c",
                                      use_existing_so="smart")

        self.assertFalse(M4_c_func.reused_c_code)

        ts4 = sp2c.get_meta_data(self.matrix_c_file_path)["timestamp"]

        md4 = sp2c.get_meta_data(self.matrix_c_file_path)

        self.assertNotEqual(ts3, ts4)

        r4 = M4_c_func(*args).flatten()
        for i in range(len(self.M1)):
            self.assertEqual(r4[i], 0)

    def test_07__boolean_expression(self):

        x3, x4 = sp.symbols("x3, x4")
        expr = Piecewise((-1, x3 < 0), (1, True))*Piecewise((0, ITE(x3 < 0, Abs(x3) < 0.3, False)), ((-Piecewise((0.3, x3 < 0), (0, True)) + Abs(x3))/(0.95 - Piecewise((0.3, x3 < 0), (0, True))), Abs(x3) < 0.95), (1, True))

        xx = np.linspace(-1, 1, 500)
        func_lmd = sp.lambdify(x3, expr)
        yy_lmd = np.array([func_lmd(x) for x in xx])

        func_c = sp2c.convert_to_c(x3, expr)
        yy_c = np.array([func_c(x) for x in xx])

        if 0:
            from matplotlib import pyplot as plt
            plt.plot(xx, yy_lmd)
            plt.plot(xx, yy_c + .1, "--")
            plt.show()
        self.assertTrue(np.allclose(yy_lmd - yy_c, 0))

    def test_08__custom_function(self):

        x1, k = sp.symbols("x1, k")

        # separate piecewise expression for better maintainability
        pw_expr = Piecewise((1.4, (Abs(0.1*k - 22.5) < 0.001) | (Abs(0.1*k - 5) < 0.001)), (0, True))

        # motivated by special use case
        def counter_start_func_imp(counter_k_start, k, counter_index_state, i, initial_value):
            return counter_k_start*2.5

        counter_start_func = implemented_function(f"counter_start_func", counter_start_func_imp)
        expr = counter_start_func(x1, k, x1, 2, 0.0790139064475348*x1*pw_expr)

        with self.assertRaises(NotImplementedError):
            func_c = sp2c.convert_to_c((x1, k), expr)


        counter_start_func.c_implementation = """

        double counter_start_func(double counter_k_start, double k, double counter_index_state, double i, double initial_value) {
           double result;
            result = counter_k_start*2.5;
            return result;
        }
        """
        expr = counter_start_func(x1, k, x1, 2, 0.0790139064475348*x1*pw_expr)

        sp2c.core.CLEANUP = False
        # sp2c.convert_to_c((x1, k), expr.args[-1])
        sp2c.convert_to_c((x1, k), expr)

        xx = np.linspace(-1, 1, 500)
        func_lmd = sp.lambdify((x1, k), expr)
        yy_lmd = np.array([func_lmd(x, 23) for x in xx])

        func_c = sp2c.convert_to_c((x1, k), expr)
        yy_c = np.array([func_c(x, 23) for x in xx])

        self.assertTrue(np.allclose(yy_lmd - yy_c, 0))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
