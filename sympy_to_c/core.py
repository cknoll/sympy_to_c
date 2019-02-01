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

# keep track of which attributes of which objects have been replaced
# (see _enable_reproducible_pickle_repr_for_expr)
# key: object; value: (attribute_name, original object)
replaced_attributes = {}


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


def compile_ccode(cfilepath):
    assert cfilepath.endswith(".c")
    objfilepath = "{}.o".format(cfilepath[:-2])
    sofilepath = "{}.so".format(cfilepath[:-2])
    cmd1 = "gcc -c -fPIC -lm {} -o {}".format(cfilepath, objfilepath)
    cmd2 = "gcc -shared {} -o {}".format(objfilepath, sofilepath)

    print(cmd1)
    assert os.system(cmd1) == 0

    print("\n{}\n".format(cmd2))
    assert os.system(cmd2) == 0

    # it might be the case that we already created a file with that name in that session
    # e.g. during unit-testing
    if not sofilepath in created_so_files:
        created_so_files.append(sofilepath)

    if CLEANUP:
        os.remove(cfilepath)
        os.remove(objfilepath)

    return sofilepath


def convert_to_c(args, expr, basename="expr", cfilepath="sp2clib.c", pathprefix=None,
                 use_exisiting_so=True, additional_metadata=None):
    """

    :param args:
    :param expr:
    :param basename:
    :param cfilepath:
    :param pathprefix:
    :param use_exisiting_so:    either True (fastest), False (most secure) or "smart" (compromise).
                                Optionally omit the generation of new c-code if an .so-file with
                                appropriate name (value `True`) or expr-hash (option `"smart"`)
                                already exists (True).
    :param additional_metadata: None or dict. Content will be stored inside the base64-coded
                                metadata

    :return:    python-callable wrapping the respective c-functions
    """

    if pathprefix is None:
        pathprefix = path_of_caller()
    assert isinstance(pathprefix, basestring)

    cfilepath = os.path.join(pathprefix, cfilepath)

    sopath = _get_so_path(cfilepath)
    if sopath in loaded_so_files:
        # ensure to use actual information
        unload_lib(sopath)
        _loadlib(sopath)

    if isinstance(expr, sp.MatrixBase):
        shape = expr.shape
        # ensure immutable type
        expr_matrix = sp.ImmutableDenseMatrix(expr)
        scalar_flag = False
    else:
        scalar_flag = True
        shape = (1, 1)
        expr_matrix = sp.ImmutableDenseMatrix([expr])

    # convert expr to pickle-string and calculate the hash
    # this is faster converting expr to str and then taking the hash
    fingerprint = reproducible_fast_hash(expr_matrix)
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
            res = load_func(sopath)
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
        shape=expr_matrix.shape
    )

    if additional_metadata is None:
        additional_metadata = {}
    assert not set(metadata.keys()).intersection(additional_metadata.keys())
    metadata.update(_dict_to_ordered_dict(additional_metadata))
    metadata_s = b64encode(pickle.dumps(metadata))

    _generate_ccode(args, expr_matrix, basename, cfilepath, shape, md=metadata_s)

    sopath = compile_ccode(cfilepath)
    sopath = ensure_valid_libpath(sopath)

    if sopath in loaded_so_files:
        # again ensure to use actual information
        unload_lib(sopath)
        _loadlib(sopath)
    res = load_func(sopath)
    res.reused_c_code = False
    res.metadata = metadata
    return res


def load_func(sopath, basename=None, scalar_flag=None, shape=None, nargs=None):

    md = get_meta_data(sopath)

    if basename is None:
        basename = "expr"
    if scalar_flag is None:
        scalar_flag = md["scalar_flag"]
    if shape is None:
        shape = md["shape"]
    if nargs is None:
        nargs = md["nargs"]

    if scalar_flag:
        funcname = _get_c_func_name(basename, 0, 0)
        return load_func_from_solib(sopath, funcname, nargs)
    else:
        return load_matrix_func_from_solib(sopath, basename, shape, nargs)


def get_meta_data(libpath, reload_lib=False):
    """
    try to load the .so file and try to call the get_meta_data() function. This returns
    a base64-encoded byte-array of a pickled dict

    :param libpath:        path of the .so- or c-file (from which the .so file was created)
    :param reload_lib:     flag that dertmines whether to reload the lib (this might break
                           references and lead to segfaults)

    :return: dict with meta data
    """

    if not libpath.endswith(".so"):
        libpath = _get_so_path(libpath)
    else:
        libpath = ensure_valid_libpath(libpath)

    # be sure to load the actual metadata
    if reload_lib and (libpath in loaded_so_files):
        unload_lib(libpath)

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

    return ensure_valid_libpath(sopath)


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

        c_res = codegen((partfuncname, tmp_expr), "C", "test",
                        header=False, empty=False, argument_sequence=args)
        [(c_name, ccode), (h_name, c_header)] = c_res

        ccode = "\n".join(line for line in ccode.split("\n") if not line.startswith("#include"))

        ccode_list.append(ccode)

    res = "\n\n".join(ccode_list)

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

    with open(libname, "w") as cfile:
        cfile.write(final_code)


def ensure_valid_libpath(libpath):
    # ensure that the path prefix is at least "./"
    prefix, name = os.path.split(libpath)
    if prefix == "":
        libpath = os.path.join(".", libpath)
    return libpath


def _loadlib(libpath):
    libpath = ensure_valid_libpath(libpath)

    if libpath in loaded_so_files:
        lib = loaded_so_files[libpath]
    else:
        try:
            lib = ct.cdll.LoadLibrary(libpath)
        except OSError as oerr:
            raise FileNotFoundError(oerr.args[0])
        loaded_so_files[libpath] = lib
        print("loading ", libpath)
    return lib


def unload_lib(libpath):
    libpath = ensure_valid_libpath(libpath)

    if not libpath in loaded_so_files:
        msg = "{} can not be unloaded because it was not loaded.".format(libpath)
        raise ValueError(msg)

    else:
        # noinspection PyProtectedMember
        handle = loaded_so_files.get(libpath)._handle
        _ = dlclose_func(handle)

        loaded_so_files.pop(libpath)


def unload_all_libs():
    for libpath, lib in list(loaded_so_files.items()):
        unload_lib(libpath)


def load_func_from_solib(libpath, funcname, nargs, raw=False):
    """

    :param libpath:
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
        return np.r_[[f(*args) for f in M_func_list]].reshape(shape)

    return M_func


# The follwing code is a workarround for https://github.com/sympy/sympy/issues/14835
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


def _dict_to_ordered_dict(thedict):
    """
    Convert classical dict to OrderedDict (sorted by keys)
    :param thedict:     dict
    :return:            OrderedDict
    """

    assert isinstance(thedict, dict)
    return OrderedDict(sorted(thedict.items()))


def _enable_reproducible_pickle_repr_for_expr(expr):
    """
    Convert all attributes which are dicts to OrderedDict. (See motivation above).
    Store the original objects for later recovery.

    :param expr:
    :return:        None
    """

    all_dicts = _find_dicts_in_obj(expr)

    for dictname in all_dicts:
        if dictname in blacklisted_dict_names:
            continue

        thedict = getattr(expr, dictname)

        # account for attributes which have been converted earlier
        if isinstance(thedict, OrderedDict):
            continue

        assert isinstance(thedict, dict)
        try:
            newordereddict = _dict_to_ordered_dict(thedict)
        except ValueError:
            IPS()
            raise SystemExit
        try:
            setattr(expr, dictname, newordereddict)
        except AttributeError as aerr:
            pass
        else:
            replaced_attributes[expr] = (dictname, thedict)


def _rewind_all_dict_replacements():
    """
    This function serves to reset all objects which where chaged by
    _reproducible_pickle_repr_for_expr() in their original state

    :return:
    """

    # noinspection PyShadowingBuiltins
    for object, (attrname, original) in list(replaced_attributes.items()):
        setattr(object, attrname, original)

        replaced_attributes.pop(object)


def reproducible_pickle_repr(expr):
    """

    :param expr:    sympy matrix (containig the relevant expression(s))
    :return:        byte-array (result of pickle.dumps)
    """

    assert len(replaced_attributes) == 0

    assert isinstance(expr, (sp.Basic, sp.MatrixBase))

    if isinstance(expr, sp.MatrixBase):
        expr = sp.ImmutableDenseMatrix(expr)

    symbols = expr.atoms(sp.Symbol)
    # _enable_reproducible_pickle_repr_for_expr(expr)

    for s in symbols:
        _enable_reproducible_pickle_repr_for_expr(s)

    pickle_dump = pickle.dumps(expr)

    _rewind_all_dict_replacements()

    return pickle_dump


def reproducible_fast_hash(expr):
    """

    :param expr:    sympy expression or list of sympy expressions
    :return:        hash-digest (aka fingerprint)
    """

    if isinstance(expr, (list, tuple)):
        pklrepr = b"\n\n".join([reproducible_pickle_repr(e) for e in expr])
    else:
        pklrepr = reproducible_pickle_repr(expr)
    return hashlib.sha256(pklrepr).hexdigest()
