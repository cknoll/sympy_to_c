import sympy as sp
import sympy_to_c as sp2c
import os
import timeit

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

print("Compiling sp2c function...")
series_sp2c = sp2c.convert_to_c([x, x0], series_expr, cfilepath="example.c", use_exisiting_so=False)

print("Timing different variants...")
time_subs = timeit.timeit("series_subs(1, 0)", globals=globals(), number=10000)
time_lambdify = timeit.timeit("series_lambdify(1, 0)", globals=globals(), number=10000)
time_sp2c = timeit.timeit("series_sp2c(1, 0)", globals=globals(), number=10000)

print("--- Results ---")
print(f"Accurate value with subs: {float(expr.subs([(x, 1), (x0, 0)]))}")
print(f"Series value with subs: {series_subs(1, 0)}, Time: {time_subs}")
print(f"Series value with lambdify: {series_lambdify(1, 0)}, Time: {time_lambdify}")
print(f"Series value with sp2c: {series_sp2c(1, 0)}, Time: {time_sp2c}")
