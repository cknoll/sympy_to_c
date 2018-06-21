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
# import dill as pickle
import pickle
import hashlib
import datetime


# handle python2 and python3
try:
    from base64 import encodebytes as b64encode
except ImportError:
    from base64 import encodestring as b64encode
try:
    from base64 import decodebytes as b64decode
except ImportError:
    from base64 import decodestring as b64decode

if sys.version_info[0] == 3:
    basestring = str

from ipydex import IPS  # just for debugging

CLEANUP = True

created_so_files = []


meta_data_template = """
const char* metadata =
"{}";
"""


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
    :param use_exisiting_so:    either True (fastest), False (most secure) or "smart" (compromise).
                                Optionally omit the generation of new c-code if an .so-file with
                                appropriate name (value `True`) or expr-hash (option `"smart"`)
                                already exists (True).

    :return:    python-callable wrapping the respective c-functions
    """

    if pathprefix is None:
        pathprefix = path_of_caller()
    assert isinstance(pathprefix, basestring)

    cfilepath = os.path.join(pathprefix, cfilepath)

    sopath = _get_so_path(cfilepath)
    if isinstance(expr, sp.MatrixBase):
        shape = expr.shape
        expr_matrix = expr
        scalar_flag = False
    else:
        scalar_flag = True
        shape = (1, 1)
        expr_matrix = sp.Matrix([expr])

    # convert expr to pickle-string and calculate the hash
    # this is faster converting expr to str and then taking the hash
    pexpr = pickle.dumps(expr_matrix)
    fingerprint = hashlib.sha256(pexpr).hexdigest()
    print(expr_matrix)
    print ("fingerprint:\n", fingerprint)
    if use_exisiting_so == "smart":
        md = get_meta_data(cfilepath)
        if md["fingerprint"] == fingerprint:
            use_exisiting_so = True
        else:
            print("Fingerprints of expression do not match.\n"
            "Regeneration of shared object.")
            use_exisiting_so = False

    if use_exisiting_so:
        if not os.path.isfile(sopath):
            print("Could not find {}. Create and compile new c-code.".format(sopath))
        else:
            res = load_func(sopath, basename, scalar_flag, expr_matrix.shape, len(args))
            res.reused_c_code = True
            return res

    metadata = dict(fingerprint=fingerprint,
                    timestamp=datetime.datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S.%f"),
                    nargs=len(args),
                    pexpr=pexpr,
                    expr=expr_matrix)
    metadata_s = b64encode(pickle.dumps(metadata))

    print(hashlib.sha256(pickle.dumps(expr_matrix)).hexdigest())

    _generate_ccode(args, expr_matrix, basename, cfilepath, shape, md=metadata_s)

    print(hashlib.sha256(pickle.dumps(expr_matrix)).hexdigest())

    sopath = compile_ccode(cfilepath)
    res = load_func(sopath, basename, scalar_flag, expr_matrix.shape, len(args))
    res.reused_c_code = False
    return res


def load_func(sopath, basename, scalar_flag, shape, nargs):
    if scalar_flag:
        funcname = _get_c_func_name(basename, 0, 0)
        return load_func_from_solib(sopath, funcname, nargs)
    else:
        return load_matrix_func_from_solib(sopath, basename, shape, nargs)


def get_meta_data(cfilepath):
    """
    try to load the .so file and try to call the get_meta_data() function. This returns
    a base64-encoded byte-array of a pickled dict

    :param cfilepath:      path of the c-file (from which the .so file was / would be created)
    :return: dict with meta data
    """

    libpath = _get_so_path(cfilepath)
    # fncname = "get_meta_data"

    lib = _loadlib(libpath)

    try:
        # load pointers
        ptr = ct.c_char_p.in_dll(lib, "metadata")
    except ValueError as err:
        msg = "The shared object has no stored meta data."
        raise AttributeError(msg)

    # dereference pointer
    md_encoded = ptr.value
    md = pickle.loads(b64decode(md_encoded))
    assert isinstance(md, dict)

    return md


def _get_so_path(cfilepath):
    assert cfilepath.endswith(".c")
    sopath = "{}.so".format(cfilepath[:-2])
    return sopath


def _generate_ccode(args, expr_matrix, basename, libname, shape, md=None):
    """

    :param args:
    :param expr_matrix:
    :param basename:
    :param libname:
    :param shape:
    :param md:              Metadata-string
    :return:
    """

    nr, nc = shape
    # list of index-pairs
    idcs = it.product(range(nr), range(nc))

    ccode_list = []
    for i, j in idcs:
        tmp_expr = expr_matrix[i, j]

        partfuncname = _get_c_func_name(basename, i, j)

        print(hashlib.sha256(pickle.dumps(expr_matrix)).hexdigest(), i, j)
        c_res = codegen( (partfuncname, tmp_expr), "C", "test",
                         header=False, empty=False, argument_sequence=args)
        print(hashlib.sha256(pickle.dumps(expr_matrix)).hexdigest(), i, j)
        [(c_name, ccode), (h_name, c_header)] = c_res

        ccode = "\n".join(line for line in ccode.split("\n") if not line.startswith("#include"))

        ccode_list.append(ccode)

    res = "\n\n".join(ccode_list)

    final_code = "#include <math.h>\n\n{}".format(res)

    print(hashlib.sha256(pickle.dumps(expr_matrix)).hexdigest())

    if md is not None:
        md1 = md.decode("ascii")
        newline = "\n"
        quoted_newline = '"\n"'

        if md1.endswith(newline):
            md1 = md1[:-1]
        md2 = md1.replace(newline, quoted_newline)

        md_var = meta_data_template.format(md2)
        final_code = "{}\n{}".format(final_code, md_var)
    print(hashlib.sha256(pickle.dumps(expr_matrix)).hexdigest())

    with open(libname, "w") as cfile:
        cfile.write(final_code)


def _loadlib(libpath):
    # ensure that the path prefix is at least "./"
    prefix, name = os.path.split(libpath)
    if prefix == "":
        libpath = os.path.join(".", libpath)
    lib = ct.cdll.LoadLibrary(libpath)

    return lib


def load_func_from_solib(libpath, funcname, nargs, raw=False):
    """

    :param libname:
    :param funcname:
    :param raw:         Boolean (default: `False`) return the unwrapped c-function
    :param nargs:       number of float args
    :return:
    """

    lib = _loadlib(libpath)

    # TODO: throw exception on failure
    the_c_func = getattr(lib, funcname)

    if raw:
        return the_c_func

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
        if not len(args) == nargs:
            msg = "invalid number of args. Got {}, but expected {}".format(len(args), nargs)
            raise ValueError(msg)
        return np.r_[ [f(*args) for f in M_func_list] ].reshape(shape)

    return M_func
