import sympy as sp
import sympy_to_c as sp2c
import numpy as np
import os
import timeit
try:
    # Run benchmark with Numba, if it is installed
    from numba import jit
    numba_loaded = True
except ImportError:
    numba_loaded = False

# Add GCC to the PATH environment variable. Modify this with your MinGW installation directory.
os.environ['PATH'] += os.pathsep + r"[YOUR_MINGW_DIR]\mingw64\bin"

# Set up the SymPy expression to evaluate later
x, x0 = sp.symbols('x x0')
expr = sp.exp(x**2+sp.cos(x))
order = 10

# Convert to a truncated Taylor series
print("Building series expression...")
series_expr = sum([1/sp.factorial(i)*sp.diff(expr, x, i).subs(x, x0)*x**i for i in range(order+1)])
print(series_expr)


# Define a function to evaluate the expression using SymPy subs()
def series_subs(x_, x0_):
    return float(series_expr.subs([(x, x_), (x0, x0_)]))


print("Lambdifying...")
series_lambdify = sp.lambdify([x, x0], series_expr, 'numpy')

if numba_loaded:
    print("Building Numba function...")
    series_numba = jit(series_lambdify, nopython=True)

print("Compiling sp2c function...")
series_sp2c = sp2c.convert_to_c([x, x0], series_expr, cfilepath="example.c")

print("Warming up different variants...")
# Generate random arguments to be used for all variants
# At least lambdify performs better with NumPy float64 arguments, so keep that in mind for a fair comparison
repeats = 100
np.random.seed(0+118+999+881+999+119+725+3)
x_args = np.random.rand(repeats)
x0_args = np.random.rand(repeats)

# The first run might take a lot longer than the following ones, so we might want to "warm up" by executing each once
# Be careful to use the same argument types here as in the timeit command (Numba compiles for specific types)
value_subs = series_subs(x_args[0], x0_args[0])
value_lambdify = series_lambdify(x_args[0], x0_args[0])
if numba_loaded:
    value_numba = series_numba(x_args[0], x0_args[0])
value_sp2c = series_sp2c(x_args[0], x0_args[0])

print("Timing different variants...")
# Use a counter 'i' to use a different random argument for each repeat, but the same arguments across the variants
time_subs = timeit.timeit("series_subs(x_args[i], x0_args[i]); i += 1", setup="i=0", globals=globals(), number=repeats)
time_lambdify = timeit.timeit("series_lambdify(x_args[i], x0_args[i]); i += 1", setup="i=0", globals=globals(), number=repeats)
if numba_loaded:
    time_numba = timeit.timeit("series_numba(x_args[i], x0_args[i]); i += 1", setup="i=0", globals=globals(), number=repeats)
time_sp2c = timeit.timeit("series_sp2c(x_args[i], x0_args[i]); i += 1", setup="i=0", globals=globals(), number=repeats)

print("--- Results ---")
print(f"Accurate value with subs: {float(expr.subs([(x, x_args[0]), (x0, x0_args[0])]))}")
print(f"Series value with subs: {value_subs}, Time: {time_subs}")
print(f"Series value with lambdify: {value_lambdify}, Time: {time_lambdify}")
if numba_loaded:
    print(f"Series value with Numba: {value_numba}, Time: {time_numba}")
print(f"Series value with sp2c: {value_sp2c}, Time: {time_sp2c}")
