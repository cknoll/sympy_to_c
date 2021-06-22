import sympy as sp
import sympy_to_c as sp2c
import os

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

series_sp2c = sp2c.convert_to_c([x, x0], series_expr, cfilepath="example.c")

print(f"Series value with subs: {float(series_expr.subs([(x, 0), (x0, 1)]))}")
print(f"Series value with sp2c: {series_sp2c(0, 1)}")
