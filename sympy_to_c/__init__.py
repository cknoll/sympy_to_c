# -*- coding: utf-8 -*-

"""Top-level package for sympy to c."""

from .core import convert_to_c, get_meta_data, unload_lib, unload_all_libs, created_so_files,\
    reproducible_fast_hash


# support leagacy imports like from sympy_to_c import sympy_to_c as sp2c
class Container(object):
    pass

sympy_to_c = Container()
sympy_to_c.convert_to_c = convert_to_c
sympy_to_c.get_meta_data = get_meta_data
sympy_to_c.unload_lib = unload_lib
sympy_to_c.unload_all_libs = unload_all_libs
sympy_to_c.created_so_files = created_so_files
sympy_to_c.reproducible_fast_hash = reproducible_fast_hash

__version__ = '0.1.2'
