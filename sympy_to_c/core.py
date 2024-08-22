# -*- coding: utf-8 -*-
"""
Created 2018-06-04 18:16:30 (based on older code)
@author: Carsten Knoll (enhancements)
"""

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
from collections import OrderedDict

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

if sys.version_info[0] == 2:
        # noinspection PyShadowingBuiltins
        FileNotFoundError = IOError
try:
    from ipydex import IPS  # just for debugging
except ImportError:
    # noinspection PyPep8Naming
    def IPS(): pass

CLEANUP = True

# serves e.g. to remove all created so-files in the tests
created_so_files = []

# keep track of which so-files have been already been loaded to enable force_reload_lib()
# format: {"<so-path>": handle}
loaded_so_files = {}

processed_implemented_functions = {}

# create a function to unload a lib
# this is from https://stackoverflow.com/a/50986803/333403
dlclose_func = ct.CDLL(None).dlclose
dlclose_func.argtypes = [ct.c_void_p]
dlclose_func.restype = ct.c_int

meta_data_template = """
const char* metadata =
"{}";
"""

blacklisted_dict_names = ["_constructor_postprocessor_mapping",]


def path_of_caller(*paths):
    frm = inspect.stack()[2]
    mod = inspect.getmodule(frm[0])
    res = os.path.dirname(mod.__file__)
    return res


def _get_c_func_name(base, i, j):
    return "{}_{}_{}".format(base, i, j)


def compile_c_code(c_file_path):
    assert c_file_path.endswith(".c")
    obj_file_path = "{}.o".format(c_file_path[:-2])
    so_file_path = "{}.so".format(c_file_path[:-2])
    cmd1 = "gcc -c -fPIC -lm {} -o {}".format(c_file_path, obj_file_path)
    cmd2 = "gcc -shared {} -o {}".format(obj_file_path, so_file_path)

    print(cmd1)
    assert os.system(cmd1) == 0

    print("\n{}\n".format(cmd2))
    assert os.system(cmd2) == 0

    # it might be the case that we already created a file with that name in that session
    # e.g. during unit-testing
    if not so_file_path in created_so_files:
        created_so_files.append(so_file_path)

    if CLEANUP:
        os.remove(c_file_path)
        os.remove(obj_file_path)

    return so_file_path


def convert_to_c(args, expr, basename="expr", c_file_path="sp2c_lib.c", pathprefix=None,
                 use_existing_so=True, additional_metadata=None):
    """

    :param args:
    :param expr:
    :param basename:
    :param c_file_path:
    :param pathprefix:
    :param use_existing_so:     either True (fastest), False (most secure) or "smart" (compromise).
                                Optionally omit the generation of new c-code if an .so-file with
                                appropriate name (value `True`) or expr-hash (option `"smart"`)
                                already exists (True).
    :param additional_metadata: None or dict. Content will be stored inside the base64-coded
                                metadata

    :return:    python-callable wrapping the respective c-functions
    """

    try:
        len(args)
    except TypeError:
        args = [args]

    if pathprefix is None:
        pathprefix = path_of_caller()
    assert isinstance(pathprefix, basestring)

    c_file_path = os.path.join(pathprefix, c_file_path)

    so_path = _get_so_path(c_file_path)
    if so_path in loaded_so_files:
        # ensure to use actual information
        unload_lib(so_path)
        _loadlib(so_path)

    scalar_flag = False
    squeeze_flag = False

    if isinstance(expr, sp.MatrixBase):
        # ensure immutable type
        expr_matrix = sp.ImmutableDenseMatrix(expr)
        shape = expr_matrix.shape
    elif isinstance(expr, (list, tuple)):
        expr_matrix = sp.ImmutableDenseMatrix(expr)
        shape = (len(expr),)
        squeeze_flag = True
    else:
        scalar_flag = True
        shape = None
        expr_matrix = sp.ImmutableDenseMatrix([expr])

    # convert expr to pickle-string and calculate the hash
    # this is faster converting expr to str and then taking the hash
    fingerprint = reproducible_fast_hash(expr_matrix)
    if use_existing_so == "smart":
        md = get_meta_data(c_file_path)
        if md["fingerprint"] == fingerprint:
            use_existing_so = True
        else:
            print("Fingerprints of expression do not match.\n"
                  "Regeneration of shared object.")
            use_existing_so = False

    if use_existing_so:
        if not os.path.isfile(so_path):
            print("Could not find {}. Create and compile new c-code.".format(so_path))
        else:
            res = load_func(so_path)
            res.reused_c_code = True
            return res

    # use OrderedDict for reproducibility
    metadata = OrderedDict(
        fingerprint=fingerprint,
        timestamp=datetime.datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S.%f"),
        nargs=len(args),
        args=args,
        # expr=expr_matrix,
        scalar_flag=scalar_flag,
        shape=expr_matrix.shape,
        squeeze_flag=squeeze_flag,
    )

    if additional_metadata is None:
        additional_metadata = {}
    assert not set(metadata.keys()).intersection(additional_metadata.keys())
    metadata.update(additional_metadata)
    metadata_s = b64encode(pickle.dumps(metadata))

    _generate_c_code(args, expr_matrix, basename, c_file_path, shape, md=metadata_s)

    so_path = compile_c_code(c_file_path)
    so_path = ensure_valid_lib_path(so_path)

    if so_path in loaded_so_files:
        # again ensure to use actual information
        unload_lib(so_path)
        _loadlib(so_path)
    res = load_func(so_path)
    res.reused_c_code = False
    res.metadata = metadata
    return res


def load_func(so_path, basename=None, scalar_flag=None, shape=None, nargs=None, squeeze_flag=None):

    md = get_meta_data(so_path)

    if basename is None:
        basename = "expr"
    if scalar_flag is None:
        scalar_flag = md["scalar_flag"]
    if squeeze_flag is None:
        squeeze_flag = md.get("squeeze_flag", False)
    if shape is None:
        shape = md["shape"]
    if nargs is None:
        nargs = md["nargs"]

    if scalar_flag:
        func_name = _get_c_func_name(basename, 0, 0)
        loaded_func =  load_func_from_so_lib(so_path, func_name, nargs)
    else:
        loaded_func = load_matrix_func_from_so_lib(so_path, basename, shape, nargs)

    if squeeze_flag:
        def final_func(*args):
            return np.squeeze(loaded_func(*args))
    else:
        final_func = loaded_func

    return final_func


def get_meta_data(lib_path, reload_lib=False):
    """
    try to load the .so file and try to call the get_meta_data() function. This returns
    a base64-encoded byte-array of a pickled dict

    :param lib_path:       path of the .so- or c-file (from which the .so file was created)
    :param reload_lib:     flag that determines whether to reload the lib (this might break
                           references and lead to segfaults)

    :return: dict with meta data
    """

    if not lib_path.endswith(".so"):
        lib_path = _get_so_path(lib_path)
    else:
        lib_path = ensure_valid_lib_path(lib_path)

    # be sure to load the actual metadata
    if reload_lib and (lib_path in loaded_so_files):
        unload_lib(lib_path)

    lib = _loadlib(lib_path)

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


def _get_so_path(c_file_path):
    assert c_file_path.endswith(".c")
    so_path = "{}.so".format(c_file_path[:-2])

    return ensure_valid_lib_path(so_path)


def _generate_c_code(args, expr_matrix, basename, libname, shape, md=None):
    """

    :param args:
    :param expr_matrix:
    :param basename:
    :param libname:
    :param shape:
    :param md:              Metadata-string
    :return:
    """

    nr, nc = expr_matrix.shape
    # list of index-pairs
    idcs = it.product(range(nr), range(nc))

    c_code_list = []
    for i, j in idcs:
        tmp_expr = expr_matrix[i, j]

        process_implemented_functions(tmp_expr)

        part_func_name = _get_c_func_name(basename, i, j)

        c_res = codegen((part_func_name, tmp_expr), "C", "test",
                        header=False, empty=False, argument_sequence=args)
        [(c_name, c_code), (h_name, c_header)] = c_res

        c_code = "\n".join(line for line in c_code.split("\n") if not line.startswith("#include"))

        c_code = convert_int_func_to_double(c_code)
        c_code = convert_booleans(c_code)

        c_code_list.append(c_code)

    res = "\n\n".join(c_code_list)

    final_code = "#include <math.h>\n\n{}".format(res)

    if md is not None:
        md1 = md.decode("ascii")
        newline = "\n"
        quoted_newline = '"\n"'

        if md1.endswith(newline):
            md1 = md1[:-1]
        md2 = md1.replace(newline, quoted_newline)

        md_var = meta_data_template.format(md2)
        final_code = "{}\n{}".format(final_code, md_var)

    with open(libname, "w") as c_file:
        c_file.write(final_code)


def process_implemented_functions(expr) -> None:
    """
    Sympy allows custom symbolic functions which can have a python implementation.

    This function deals with converting those object to C (if they have a certain attributes)
    """

    custom_functions = expr.atoms(sp.core.function.AppliedUndef)
    for cf in custom_functions:
        _process_implemented_function(cf)

def _process_implemented_function(applied_func_obj):
    func_obj = type(applied_func_obj)

    if c_implementation := getattr(func_obj, "c_implementation", None) is None:
        msg = f"Applied function of type `{func_obj.name}` without specified C implementation"
        raise NotImplementedError(msg)


def convert_booleans(c_code):

    new_c_code = c_code.replace(" true", " 1").replace(" false", " 0")
    return new_c_code


def convert_int_func_to_double(c_code):
    if c_code.startswith("double "):
        return c_code

    lines = c_code.split("\n")

    int_beginning = "int expr_"
    line0 = lines[0]
    assert line0.startswith(int_beginning)
    assert line0.count(int_beginning) == 1
    lines[0] = line0.replace(int_beginning, "double expr_")

    return_line = lines[-3]

    return_src_beginning = "return expr"
    assert return_line.count(return_src_beginning) == 1
    lines[-3] = return_line.replace(return_src_beginning, "return (double)expr")

    new_c_code = "\n".join(lines)

    return new_c_code


def ensure_valid_lib_path(lib_path):
    # ensure that the path prefix is at least "./"
    prefix, name = os.path.split(lib_path)
    if prefix == "":
        lib_path = os.path.join(".", lib_path)
    return lib_path


def _loadlib(lib_path):
    lib_path = ensure_valid_lib_path(lib_path)

    if lib_path in loaded_so_files:
        lib = loaded_so_files[lib_path]
    else:
        try:
            lib = ct.cdll.LoadLibrary(lib_path)
        except OSError as os_err:
            raise FileNotFoundError(os_err.args[0])
        loaded_so_files[lib_path] = lib
        print("loading ", lib_path)
    return lib


def unload_lib(lib_path):
    lib_path = ensure_valid_lib_path(lib_path)

    if not lib_path in loaded_so_files:
        msg = "{} can not be unloaded because it was not loaded.".format(lib_path)
        raise ValueError(msg)

    else:
        # noinspection PyProtectedMember
        handle = loaded_so_files.get(lib_path)._handle
        _ = dlclose_func(handle)

        loaded_so_files.pop(lib_path)


def unload_all_libs():
    for lib_path, lib in list(loaded_so_files.items()):
        unload_lib(lib_path)


def load_func_from_so_lib(lib_path, func_name, nargs, raw=False):
    """

    :param lib_path:
    :param func_name:
    :param raw:         Boolean (default: `False`) return the unwrapped c-function
    :param nargs:       number of float args
    :return:
    """

    lib = _loadlib(lib_path)

    # TODO: throw exception on failure
    the_c_func = getattr(lib, func_name)

    if raw:
        return the_c_func

    # this converts the result in a python float obj:
    the_c_func.restype = ct.c_double
    the_c_func.argtypes = [ct.c_double]*nargs

    # the caller must take care of the number of args
    def the_func(*args):
        assert len(args) == nargs

        res = the_c_func(*args)

        return res

    return the_func


# noinspection PyPep8Naming
def load_matrix_func_from_so_lib(libname, basename, shape, nargs):
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
        func_name = _get_c_func_name(basename, i, j)
        M_func_list.append(load_func_from_so_lib(libname, func_name, nargs))

    def M_func(*args):
        if not len(args) == nargs:
            msg = "invalid number of args. Got {}, but expected {}".format(len(args), nargs)
            raise ValueError(msg)
        return np.r_[[f(*args) for f in M_func_list]].reshape(shape)

    return M_func


# The following code is a workaround for https://github.com/sympy/sympy/issues/14835
# It serves to generate a reproducible pickle representation of sympy expressions
# Original pickle representation may vary due to dict sorting depending on builtin hash()
# which is randomized for security reasons

def _find_dicts_in_obj(obj):
    """
    Cycle through all attributes of obj and return those attribute-names whose type is dict.
    To avoid unnecessary work, the result is cached as an attribute of the sympy-module-object
    :return:    list of strings
    """

    if not hasattr(sp, "_obj_dict_attrbs"):
        sp._obj_dict_attrbs = dict()

    # noinspection PyUnresolvedReferences, PyProtectedMember
    res = sp._obj_dict_attrbs.get(type(obj))

    if res is not None:
        assert isinstance(res, list)
        return res

    # nothing was found in cache -> we have to inspect the obj

    all_dicts = []

    for a in dir(obj):
        try:
            class_attr = getattr(type(obj), a)
            if isinstance(class_attr, property):
                continue

            if isinstance(getattr(obj, a), dict):
                all_dicts.append(a)
        except AttributeError:
            pass

    # set the cache
    # noinspection PyUnresolvedReferences, PyProtectedMember
    sp._obj_dict_attrbs[type(obj)] = all_dicts

    return all_dicts

def reproducible_pickle_repr(expr) -> bytes:
    """

    :param expr:    sympy matrix (containing the relevant expression(s))
    :return:        byte-array (result of pickle.dumps)
    """

    # in the past this function had to do much more

    assert isinstance(expr, (sp.Basic, sp.MatrixBase))

    if isinstance(expr, sp.MatrixBase):
        expr = sp.ImmutableDenseMatrix(expr)

    try:
        pickle_dump = pickle.dumps(expr)
    except pickle.PickleError:
        # TODO: print warning (e.g. long runtime of str. representation)
        # TODO test dill with custom functions
        pickle_dump = repr(expr).encode("utf8")

    return pickle_dump


def reproducible_fast_hash(expr):
    """

    :param expr:    sympy expression or list of sympy expressions
    :return:        hash-digest (aka fingerprint)
    """

    if isinstance(expr, (list, tuple)):
        pkl_repr = b"\n\n".join([reproducible_pickle_repr(e) for e in expr])
    else:
        pkl_repr = reproducible_pickle_repr(expr)
    return hashlib.sha256(pkl_repr).hexdigest()
