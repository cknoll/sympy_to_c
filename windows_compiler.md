# Installing GCC on Windows

This package requires GCC to be installed to compile the C code that is generated from SymPy expressions to a dynamic library which can then be loaded in Python. On Linux almost every distribution typically includes GCC, so the package should work out of the box there. On Windows you will need to install and configure it yourself. To do that, you can follow this guide.

## Choosing and downloading the compiler tools

There are multiple projects that make GCC available on Windows:
* [MinGW-w64](https://mingw-w64.org/doku.php) **Recommended**
* [MinGW](http://www.mingw.org/)
* [Cygwin](https://www.cygwin.com/)

We recommend **MinGW-w64**, which is also the only one we have tested and will document here. Only use another option, if you have specific reasons for it.

## Installing MinGW-w64

Download a MinGW build from [the official website](https://mingw-w64.org/doku.php/download).
The easiest option will probably be **MinGW-w64-builds**.
Then execute the installer and choose the *Architecture* depending on what kind of Python installation you're using:
* if you use Python **32 bit**, choose **i686**,
* if you use Python **64 bit**, choose **x86_64**.

Leave the other options as is and complete the installation.

## Adding compiler binaries to PATH

For sympy_to_c to be able to find GCC, it needs to be in the PATH environment variable.
There are multiple ways of achieving that, the simplest is adding it globally, if you don't have another GCC installation used by other software.
Alternatively you can configure it locally only when using sympy_to_c.

### Adding GCC to system PATH

Open *Control Panel*, select *Users* and then *Edit environment variables*.
If you want this GCC installation to be used by all users, select 'Path' in the *system variables* section, otherwise select it in *user variables* and click *Edit*.
The directory you will need to add is `[YOUR_MINGW_PATH]\mingw64\bin\`, the directory must contain `gcc.exe`.

### Modifying the PATH for the current script

If you don't want to modify the global PATH, you can do it only for the Python script you want to execute.
Use the `os` module to change the environment variable only for the current process, which leaves it unchanged after the process exits.

```Python
import os

os.environ['PATH'] += os.pathsep + r"[YOUR_MINGW_PATH]\mingw64\bin"
```
