#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sympy_to_c` package."""


import unittest
import sympy as sp
import numpy as np
import os
import sys
import hashlib
import pickle
import copy

from ipydex import IPS

from sympy_to_c import sympy_to_c as sp2c

if sys.version_info[0] == 2:
    input = raw_input


# noinspection PyPep8Naming, PyTypeChecker
class TestSympy_to_c(unittest.TestCase):
    """Tests for `sympy_to_c` package."""

    __name__ = "test"

    def setUp(self):
        """Set up test fixtures, if any."""
        x1, x2, x3 = self.xx = sp.symbols("x1, x2, x3")

        self.e1 = x1*x2 + x3

        self.M1 = sp.Matrix([self.e1, sp.sin(x1)*sp.exp(x3), 42])

        np.random.seed(1840)
        N = 100
        self.XX = np.random.random((N, len(self.xx)))

    def tearDown(self):
        """Tear down test fixtures, if any."""
        while sp2c.created_so_files:
            sofilepath = sp2c.created_so_files.pop()
            os.remove(sofilepath)

    def test_scalar_expression(self):
        """Test conversion of simple scalar expression."""

        e1_c_func = sp2c.convert_to_c(self.xx, self.e1, cfilepath="scalar.c",
                                      use_exisiting_so=False)
        e1_l_func = sp.lambdify(self.xx, self.e1)

        for xx in self.XX:
            self.assertAlmostEqual(e1_c_func(*xx), e1_l_func(*xx))

    def test_matrix_expression(self):
        """Test conversion of simple matrix."""

        M1_c_func = sp2c.convert_to_c(self.xx, self.M1, cfilepath="matrix.c",
                                      use_exisiting_so=False)
        M1_l_func = sp.lambdify(self.xx, self.M1)

        for xx in self.XX:
            self.assertTrue(np.allclose(M1_c_func(*xx), M1_l_func(*xx)))


    def test_meta_data(self):

        # create new so-file
        sp2c.CLEANUP = False
        M1_c_func = sp2c.convert_to_c(self.xx, self.M1, cfilepath="matrix.c",
                                      use_exisiting_so=False)

        md = sp2c.get_meta_data("matrix.c")

        self.assertTrue(isinstance(md, dict))
        self.assertTrue("fingerprint" in md)
        self.assertTrue("timestamp" in md)
        self.assertTrue("nargs" in md)

        self.assertEqual(md["nargs"], len(self.xx))

    @unittest.expectedFailure
    def test_hashing1(self):

        # e1 = self.M1[0]
        e1 = sp.symbols("x1")
        # print(pickle.dumps(e1))

        h1 = hashlib.sha256(pickle.dumps(e1)).hexdigest()
        M1_1 = self.M1.copy()
        h2 = hashlib.sha256(pickle.dumps(e1)).hexdigest()



        from sympy.utilities.codegen import codegen

        print(hashlib.sha256(pickle.dumps(e1)).hexdigest())

        if 0:
            M1_c_func = sp2c.convert_to_c(self.xx, e1, cfilepath="matrix.c",
                                          use_exisiting_so=False)
        else:

            # this call changes something in the pickle representation
            codegen(("M1_00", e1), "C", "test",
                             header=False, empty=False, argument_sequence=self.xx)

        # print("..")
        # print(pickle.dumps(e1))

        h3 = hashlib.sha256(pickle.dumps(e1)).hexdigest()

        self.assertEqual(h1, h2)
        self.assertEqual(h1, h3)

    def test_hashing2(self):

        e1 = sp.symbols("x1", integer=False)
        e2 = copy.deepcopy(e1)
        e3 = sp.Symbol(e1.name, **e1._assumptions)

        xx_a = [e1]

        h1a = hashlib.sha256(pickle.dumps(e1)).hexdigest()
        # h1b = hashlib.sha256(pickle.dumps(e2)).hexdigest()
        # h1c = hashlib.sha256(pickle.dumps(e3)).hexdigest()

        from sympy.utilities.codegen import codegen
        codegen(("M1_00", e1), "F95", "M1_00",
               header=False, empty=False, argument_sequence=xx_a)

        h2a = hashlib.sha256(pickle.dumps(e1)).hexdigest()
        # h2b = hashlib.sha256(pickle.dumps(e2)).hexdigest()
        # h2c = hashlib.sha256(pickle.dumps(e3)).hexdigest()

        w1 = sp.Symbol(e1.name, **e1.assumptions0)
        h3 = hashlib.sha256(pickle.dumps(w1)).hexdigest()


        # print("\n".join([h1a, h1b, h1c, h2a, h2b, h2c, h3]))
        print("\n".join([h1a, h2a,]))

        # IPS()


    # @unittest.expectedFailure
    def test_use_existing_old(self):

        # avoid some strange interation between pickle and sympy
        M1_1 = self.M1.copy()
        M1_2 = self.M1.copy()
        M1_3 = self.M1.copy()
        M1_4 = self.M1.copy()

        xx1 = copy.deepcopy(self.xx)
        xx2 = copy.deepcopy(self.xx)
        xx3 = copy.deepcopy(self.xx)
        xx4 = copy.deepcopy(self.xx)


        """
        For some currently unknown reason the pickle-representation of some sympy objects
        changes after they are copied or after they are put into convert_to_c
        
        """

        # create new so-file
        sp2c.CLEANUP = False

        # print(hashlib.sha256(pickle.dumps(xx1)).hexdigest())
        M1_c_func = sp2c.convert_to_c(xx1, M1_1, cfilepath="matrix.c",
                                      use_exisiting_so=False)

        # other expression but no new c-Code
        M2_c_func = sp2c.convert_to_c(xx2, M1_2*0, cfilepath="matrix.c",
                                      use_exisiting_so=True)

        self.assertTrue(M2_c_func.reused_c_code)

        # test that result is the same
        args = (1702, 123, -12.34)
        r1 = M1_c_func(*args).flatten()
        r2 = M2_c_func(*args).flatten()
        for i in range(len(self.M1)):
            self.assertEqual(r1[i], r2[i])

        ts2 = sp2c.get_meta_data("matrix.c")["timestamp"]

        # same expression -> no new code
        print(3)
        input(3)
        M3_c_func = sp2c.convert_to_c(xx3, M1_3, cfilepath="matrix.c",
                                      use_exisiting_so="smart")

        ts3 = sp2c.get_meta_data("matrix.c")["timestamp"]
        if 0:
            # this should work
            self.assertTrue(M3_c_func.reused_c_code)

            self.assertEqual(ts2, ts3)

        input(4)

        M4_c_func = sp2c.convert_to_c(xx4, M1_4*0, cfilepath="matrix.c",
                                      use_exisiting_so="smart")

        self.assertFalse(M4_c_func.reused_c_code)

        ts4 = sp2c.get_meta_data("matrix.c")["timestamp"]
        self.assertNotEqual(ts3, ts4)

        r4 = M4_c_func(*args).flatten()
        for i in range(len(self.M1)):
            self.assertEqual(r4[i], 0)

    # @unittest.expectedFailure
    def test_use_existing(self):

        # avoid some strange interation between pickle and sympy
        M1_1 = self.M1.copy()
        M1_2 = self.M1.copy()
        M1_3 = self.M1.copy()
        M1_4 = self.M1.copy()

        xx1 = copy.deepcopy(self.xx)
        xx2 = copy.deepcopy(self.xx)
        xx3 = copy.deepcopy(self.xx)
        xx4 = copy.deepcopy(self.xx)


        """
        For some currently unknown reason the pickle-representation of some sympy objects
        changes after they are copied or after they are put into convert_to_c
        
        """

        # create new so-file
        sp2c.CLEANUP = False

        # print(hashlib.sha256(pickle.dumps(xx1)).hexdigest())
        M1_c_func = sp2c.convert_to_c(xx1, M1_1, cfilepath="matrix.c",
                                      use_exisiting_so=False)

        # other expression but no new c-Code
        M2_c_func = sp2c.convert_to_c(xx2, M1_2*0, cfilepath="matrix.c",
                                      use_exisiting_so=True)

        self.assertTrue(M2_c_func.reused_c_code)

        # test that result is the same
        args = (1702, 123, -12.34)
        r1 = M1_c_func(*args).flatten()
        r2 = M2_c_func(*args).flatten()
        for i in range(len(self.M1)):
            self.assertEqual(r1[i], r2[i])

        ts2 = sp2c.get_meta_data("matrix.c")["timestamp"]

        # same expression -> no new code
        print(3)
        input(3)
        M3_c_func = sp2c.convert_to_c(xx3, M1_3, cfilepath="matrix.c",
                                      use_exisiting_so="smart")

        ts3 = sp2c.get_meta_data("matrix.c")["timestamp"]
        if 0:
            # this should work
            self.assertTrue(M3_c_func.reused_c_code)

            self.assertEqual(ts2, ts3)

        input(4)

        M4_c_func = sp2c.convert_to_c(xx4, M1_4*0, cfilepath="matrix.c",
                                      use_exisiting_so="smart")

        self.assertFalse(M4_c_func.reused_c_code)

        ts4 = sp2c.get_meta_data("matrix.c")["timestamp"]
        self.assertNotEqual(ts3, ts4)

        r4 = M4_c_func(*args).flatten()
        for i in range(len(self.M1)):
            self.assertEqual(r4[i], 0)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
