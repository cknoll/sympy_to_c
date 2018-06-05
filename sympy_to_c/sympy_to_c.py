# -*- coding: utf-8 -*-
from __future__ import print_function

import ctypes as ct

import inspect
import os
import numpy as np
import itertools as it
import sympy as sp
from sympy.utilities.codegen import codegen
import sys

if sys.version_info[0] == 3:
    basestring = str

from ipydex import IPS  # just for debugging

CLEANUP = True

created_so_files = []


def path_of_caller(*paths):
    frm = inspect.stack()[2]
    mod = inspect.getmodule(frm[0])
    res = os.path.dirname(mod.__file__)
    return res


def _get_c_func_name(base, i, j):
    return "{}_{}_{}".format(base, i, j)


def compile_ccode(cfilepath):

    assert cfilepath.endswith(".c")
    objfilepath = "{}.o".format(cfilepath[:-2])
    sofilepath = "{}.so".format(cfilepath[:-2])
    cmd1 = "gcc -c -fPIC -lm {} -o {}".format(cfilepath, objfilepath)
    cmd2 = "gcc -shared {} -o {}".format(objfilepath, sofilepath)

    print(cmd1)
    assert os.system(cmd1) == 0

    print("\n\n{}".format(cmd2))
    assert os.system(cmd2) == 0

    created_so_files.append(sofilepath)

    if CLEANUP:
        os.remove(cfilepath)
        os.remove(objfilepath)

    return sofilepath


def convert_to_c(args, expr, basename="expr", cfilepath="sp2clib.c", pathprefix=None,
                 use_exisiting_so=True):
    """

    :param args:
    :param expr:
    :param basename:
    :param cfilepath:
    :param use_exisiting_so:    Omits generation of c-code if an .so-file with appropriate name
                                already exists.

    :return:    python-callable wrapping the respective c-functions
    """

    if pathprefix is None:
        pathprefix = path_of_caller()
    assert isinstance(pathprefix, basestring)

    cfilepath = os.path.join(pathprefix, cfilepath)

    assert cfilepath.endswith(".c")
    sopath = "{}.so".format(cfilepath[:-2])
    if isinstance(expr, sp.MatrixBase):
        shape = expr.shape
        expr_matrix = expr
        scalar_flag = False
    else:
        scalar_flag = True
        shape = (1, 1)
        expr_matrix = sp.Matrix([expr])

    if not use_exisiting_so or not os.path.isfile(sopath):
        _generate_ccode(args, expr_matrix, basename, cfilepath, shape)

        sopath = compile_ccode(cfilepath)

    if scalar_flag:
        funcname = _get_c_func_name(basename, 0, 0)
        return load_func_from_solib(sopath, funcname, len(args))
    else:
        return load_matrix_func_from_solib(sopath, basename, expr_matrix.shape, len(args))


def _generate_ccode(args, expr_matrix, basename, libname, shape):

    nr, nc = shape
    # list of index-pairs
    idcs = it.product(range(nr), range(nc))

    ccode_list = []
    for i, j in idcs:
        tmp_expr = expr_matrix[i, j]

        partfuncname = _get_c_func_name(basename, i, j)

        c_res = codegen( (partfuncname, tmp_expr), "C", "test",
                         header=False, empty=False, argument_sequence=args)
        [(c_name, ccode), (h_name, c_header)] = c_res

        ccode = "\n".join(line for line in ccode.split("\n") if not line.startswith("#include"))

        ccode_list.append(ccode)

    res = "\n\n".join(ccode_list)

    final_code = "#include <math.h>\n\n{}".format(res)

    with open(libname, "w") as cfile:
        cfile.write(final_code)


def load_func_from_solib(libpath, funcname, nargs):
    """

    :param libname:
    :param funcname:
    :param nargs:       number of float args
    :return:
    """

    # ensure that the path prefix is at least "./"
    prefix, name = os.path.split(libpath)
    if prefix == "":
        libpath = os.path.join(".", libpath)
    lib = ct.cdll.LoadLibrary(libpath)

    # TODO: throw exception on failure
    the_c_func = getattr(lib, funcname)

    # this converts the result in a python float obj:
    the_c_func.restype = ct.c_double
    the_c_func.argtypes = [ct.c_double]*nargs

    # the caller must take care of the number of args
    def thefunc(*args):
        assert len(args) == nargs

        res = the_c_func(*args)

        return res

    return thefunc


# noinspection PyPep8Naming
def load_matrix_func_from_solib(libname, basename, shape, nargs):
    """

    :param libname:
    :param basename:
    :param shape:
    :param nargs:
    :return:
    """

    nr, nc = shape

    # list of index-pairs
    idcs = it.product(range(nr), range(nc))

    M_func_list = []
    for i, j in idcs:
        funcname = _get_c_func_name(basename, i, j)
        M_func_list.append(load_func_from_solib(libname, funcname, nargs))

    def M_func(*args):
        assert len(args) == nargs
        return np.r_[ [f(*args) for f in M_func_list] ].reshape(shape)

    return M_func
