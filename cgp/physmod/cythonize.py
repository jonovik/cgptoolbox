"""Cythonize the generated Python code for a CellML model."""
import re # Used to replace certain functions with Cython versions

from ..utils import codegen

def cythonize_model(s, modelname=""):
    """
    Cythonize the generated Python code for a CellML model.
    
    :param str s: original Python source code
    :param str modelname: name of model
    :rtype str: Cython source code
    """
    # declare constants as float to avoid integer division of constants
    s = s.replace(
        "constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;", 
        "constants = np.zeros(sizeConstants, dtype=ftype); "
        "states = np.zeros(sizeStates, dtype=ftype)")
    # specify type of vector lengths
    for i in "Algebraic", "States", "Constants":
        s = s.replace("\nsize%s = " % i, "\ncpdef int size%s = " % i)
    # Names of functions to be replaced by Cython versions
    w = "equal less greater less_equal greater_equal".split()
    # prepend imports, type declarations, and auxiliary Cython functions
    s = s.replace("\nfrom math import *\n", '''
## BEGIN Added by cythonize_model() ##

import ctypes
cimport numpy as np
import numpy as np
from numpy import *
from pysundials.cvode import NVector

ftype = np.float64 # explicit type declaration, can be used with cython
ctypedef np.float64_t dtype_t

cdef extern from "math.h":
    dtype_t log(dtype_t x)
    dtype_t exp(dtype_t x)
    dtype_t floor(dtype_t x)
    dtype_t fabs(dtype_t x)

cdef extern from "Python.h":
    ctypedef struct PyObject
    void* PyLong_AsVoidPtr(PyObject *pylong)

# C pointers to array of dtype_t
cdef inline dtype_t* bufarr(x):
    return <dtype_t*>(<np.ndarray>x).data

cdef inline dtype_t* bufnv(v):
    cdef long lp # 64-bit integers for pointers to buffers
    lp = ctypes.addressof(v.cdata.contents)
    return <dtype_t*>lp

# Numpy arrays, can be used from Python
y0 = np.zeros(sizeStates, dtype=ftype)
ydot = np.zeros(sizeStates, dtype=ftype)
p = np.zeros(sizeConstants, dtype=ftype)
algebraic = np.zeros(sizeAlgebraic, dtype=ftype)

# Pointers to array of dtype_t, fast access from Cython
cdef dtype_t* py0 = bufarr(y0)
cdef dtype_t* pydot = bufarr(ydot)
cdef dtype_t* pp = bufarr(p)
cdef dtype_t* palgebraic = bufarr(algebraic)

cpdef int ode(dtype_t t, y, ydot, f_data):
    cdef dtype_t *py, *pydot # pointers to buffers
    cdef str msg = "Use of f_data not implemented; use global array p instead"
    assert f_data is None, msg
    # make this work with both numpy.ndarray and pysundials.cvode.NVector
    if isinstance(y, NVector):
        py = bufnv(y)
    else:
        assert isinstance(y, np.ndarray)
        py = bufarr(y)
    if isinstance(ydot, NVector):
        pydot = bufnv(ydot)
    else:
        assert isinstance(ydot, np.ndarray)
        pydot = bufarr(ydot)
    # NVector has no .fill method, so do this the hard way
    cdef int i
    # ydot.fill(0.0)
    for i in range(sizeStates):
        pydot[i] = 0.0
    # algebraic.fill(0.0)
    for i in range(sizeAlgebraic):
        palgebraic[i] = 0.0
    compute_rates(t, py, pydot, pp, palgebraic)
    return 0

def rates_and_algebraic(np.ndarray[dtype_t, ndim=1] t, y):
    """
    Compute rates and algebraic variables for a given state trajectory.
    
    Unfortunately, the CVODE machinery does not offer a way to return rates and 
    algebraic variables during integration. This function re-computes the rates 
    and algebraics at each time step for the given state.
    
    >>> from cgp.physmod.cellmlmodel import Cellmlmodel
    >>> workspace = "bondarenko_szigeti_bett_kim_rasmusson_2004"
    >>> bond = Cellmlmodel(workspace, t=[0, 20])
    >>> with bond.autorestore():
    ...     bond.yr.V = 100 # simulate stimulus
    ...     t, y, flag = bond.integrate()
    >>> ydot, alg = bond.model.rates_and_algebraic(t, y)
    >>> from pylab import * # doctest: +SKIP
    >>> plot(t, alg.view(bond.dtype.a)["J_xfer"], '.-', t, y.Cai, '.-') # doctest: +SKIP
    
    Verify that this Cython version is equivalent to the pure Python version.
    
    >>> bondp = Cellmlmodel(workspace, t=[0, 20], 
    ...     use_cython=False, purge=True)
    >>> ydotp, algp = bondp.model.rates_and_algebraic(t, y)
    >>> np.testing.assert_almost_equal(ydot, ydotp, decimal=5)
    >>> np.testing.assert_almost_equal(alg, algp, decimal=5)
    """
    cdef int imax = len(t)
    y = y.view(ftype)
    ydot = np.zeros_like(y)
    alg = np.zeros((imax, len(algebraic)))
    cdef double* py
    cdef double* pydot
    cdef double * palgebraic
    cdef int i
    for i in range(imax):
        py = bufarr(y[i])
        pydot = bufarr(ydot[i])
        palgebraic = bufarr(alg[i])
        compute_rates(t[i], py, pydot, pp, palgebraic)
        compute_algebraic(t[i], py, pp, palgebraic)
    return ydot, alg


## END Added by cythonize_model() ##
''')
    # make compute_rates() a cythonized version of computeRates()
    s0 = """
def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
"""
    i0 = s.find(s0)
    s1 = """
    return(rates)
"""
    i1 = s.find(s1)
    compute_rates_code = s[i0:i1]
    # Get rid of custom_piecewise()
    L = [repcp(line) for line in compute_rates_code.split("\n")]
    # Replace some functions with Cython replacements
    L = [prepend("cy_", w, line) for line in L]
    compute_rates_code = "\n".join(L)
    s += compute_rates_code.replace(s0.strip(), """

## BEGIN Added by cythonize_model() ##

cdef inline bint cy_equal(dtype_t x, dtype_t y):
    return x == y

cdef inline bint cy_greater(dtype_t x, dtype_t y):
    return x > y

cdef inline bint cy_less(dtype_t x, dtype_t y):
    return x < y

cdef inline bint cy_greater_equal(dtype_t x, dtype_t y):
    return x >= y

cdef inline bint cy_less_equal(dtype_t x, dtype_t y):
    return x <= y

cimport cython
@cython.cdivision(True)
cdef void compute_rates(dtype_t voi, dtype_t* states, dtype_t* rates, dtype_t* constants, dtype_t* algebraic):
    pass  # in case function body is empty
""") + "\n"


    # make compute_algebraic() a cythonized version of computeAlgebraic()
    s0 = """
def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
"""
    i0 = s.find(s0)
    s1 = """
    return algebraic
"""
    i1 = s.find(s1)
    compute_algebraic_code = s[i0:i1]
    # Get rid of custom_piecewise()
    L = [repcp(line) for line in compute_algebraic_code.split("\n")]
    # Replace some functions with Cython replacements
    L = [prepend("cy_", w, line) for line in L]
    compute_algebraic_code = "\n".join(L) + "\n"
    s += compute_algebraic_code.replace(s0, """

@cython.cdivision(True)
cdef void compute_algebraic(dtype_t voi, dtype_t* states, dtype_t* constants, dtype_t* algebraic):
    pass # in case there is no function body left after eliminating s0
""") + "\n"


    s += '''
def bench():
    """
    Benchmark ode().
    
    To profile execution, prepend this line to the module::
    
        # cython: profile=True
    
    This can be used with the usual Python or IPython profiler.
    Unfortunately, Cython does not have line profiling.
    
    In IPython::
    
        >>> import module_name as m         # doctest: +SKIP
        >>> prun m.bench()                  # doctest: +SKIP
        
        Sample output:
        1400004 function calls in 9.270 CPU seconds
        ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        100000    4.952    0.000    7.322    0.000 BL6WT_200410.pyx:736(compute_rates)
        100000    1.182    0.000    8.976    0.000 BL6WT_200410.pyx:60(ode)
        400000    0.946    0.000    0.946    0.000 BL6WT_200410.pyx:731(cy_custom_piecewise)
    """
    cdef int i
    for i in range(10000):
        ode(0, y0, ydot, None)

y0[:], p[:] = initConsts()

## END Added by cythonize_model() ##
'''
    # Percent literals must be doubled when using string interpolation,
    # e.g. %%M%% to get %M%
    setup = '''
"""
%(modelname)s setup file.
Usage: python setup.py build_ext --inplace

It may be necessary to remove intermediate files from previous builds:

Linux:
rm -rf build m.so m.c && python setup.py build_ext --inplace && python -c "import m"

Windows:
del /q build m.pyd m.c
python setup.py build_ext --inplace
python -c "import m"
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import platform
import os

extname = "cy"
HOME = os.environ["HOME"]

if platform.system() == "Windows":
    ext_modules = [Extension(extname, [extname + ".pyx"],
        include_dirs=['c:/MinGW/msys/1.0/local/include', 'c:/msys/1.0/local/include', np.get_include()],
        library_dirs=['c:/MinGW/msys/1.0/local/lib', 'c:/msys/1.0/local/lib'],
        libraries=['sundials_cvode', 'sundials_nvecserial'])]
elif platform.system() == "Linux":
    if "stallo" in platform.node():
        ext_modules = [Extension(extname, [extname + ".pyx"],
            include_dirs=[HOME + '/usr/include', np.get_include()],
            library_dirs=[HOME + '/usr/lib'],
            libraries=['sundials_cvode', 'sundials_nvecserial'])]
    else: # Titan
        ext_modules = [Extension(extname, [extname + ".pyx"],
            include_dirs=[HOME + '/usr/include', '/site/VERSIONS/sundials-2.3.0/include', np.get_include()],
            library_dirs=[HOME + '/usr/lib', '/site/VERSIONS/sundials-2.3.0/lib'],
            libraries=['sundials_cvode', 'sundials_nvecserial'])]
elif platform.system() == "Darwin":  # Mac OS X
    ext_modules = [Extension(extname, [extname + ".pyx"],
        include_dirs=['/usr/local/include', np.get_include()],
        library_dirs=['/usr/local/lib'],
        libraries=['sundials_cvode', 'sundials_nvecserial'])]

setup(
    name = extname,
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)
'''
    return s, setup % dict(modelname=modelname)

def rep(s, old, new):
    """
    Replace occurrences of old([...]) with new(...) throughout s.
    
    Removing brackets and changing a function name.
    
    >>> s = "This test([includes a nested test([like this])])"
    >>> rep(s, "test", "X")
    'This X(includes a nested X(like this))'
    
    Removing brackets without changing the function name.
    
    >>> rep(s, "test", "test")
    'This test(includes a nested test(like this))'
    """
    pattern = old + "(["
    while pattern in s:
        # find the first occurrence of pattern
        before, _sep, after = s.partition(pattern)
        # find the matching closing bracket
        nesting_level = 1
        for pos, char in enumerate(after):
            if char == "[":
                nesting_level += 1
            elif char == "]":
                nesting_level -= 1
            if nesting_level == 0:
                break
        assert nesting_level == 0, "Matching bracket not found in '%s'" % s
        # remove the matching closing bracket
        after = after[:pos] + after[pos+1:]  # pylint:disable=W0631
        s = before + new + "(" + after
    return s

def cp2cond(s):
    """
    Make a conditional expression to replace a call to custom_piecewise().
    
    The CellML code generator outputs "ternary if" expressions for C 
    (cond ? val_if_true : val_if_false)
    but not (val_if_true if cond else val_if_false) for Python.
    Instead, it defines a custom_piecewise() function which is about 1000 times 
    slower and barely legible. This function returns equivalent code that uses 
    Python's native conditional expressions, which Cython can convert to 
    efficient C.
    
    The typical case is one condition and 0 otherwise.
    CellML then ends the list with True, 0.
    The final "else 0" mimics the behaviour of custom_piecewise() and 
    np.select() if no conditions match.
    
    >>> cp2cond('custom_piecewise([x<3, "<3", True, 0])')
    "('<3' if (x < 3) else 0 if True else 0)"
    
    Two conditions, so the final "else 0" is not redundant.
    
    >>> cp2cond('custom_piecewise([x<3, "<3", x>5, ">5"])')
    "('<3' if (x < 3) else '>5' if (x > 5) else 0)"
    
    Two conditions and otherwise 0, as CellML might code it.
    
    >>> cp2cond('custom_piecewise([x<3, "<3", x>5, ">5", True, 0])')
    "('<3' if (x < 3) else '>5' if (x > 5) else 0 if True else 0)"
    
    Function call as list item.
    
    >>> cp2cond('custom_piecewise([f(x,1), "<3", x>5, ">5", True, 0])')
    "('<3' if f(x, 1) else '>5' if (x > 5) else 0 if True else 0)"
    """
    p = codegen.parse(s)
    L = [codegen.to_source(i) for i in p.body[0].value.args[0].elts]
    cond = L[0::2]
    val = L[1::2]
    ifelse = " ".join("%s if %s else" % (v, c) for c, v in zip(cond, val))
    return "(%s 0)" % ifelse

def repcp(s):
    """
    Replace custom_piecewise() with conditional expressions throughout s.
    
    >>> s = "A custom_piecewise([a, b, c, d]) B custom_piecewise([e, f]) C"
    >>> repcp(s)
    'A (b if a else d if c else 0) B (f if e else 0) C'
    """
    pattern = "custom_piecewise(["
    while pattern in s:
        # find the first occurrence of pattern
        _before, sep, after = s.partition(pattern)
        # find the matching closing bracket
        nesting_level = 1
        for pos, char in enumerate(after):
            if char == "[":
                nesting_level += 1
            elif char == "]":
                nesting_level -= 1
            if nesting_level == 0:
                break
        assert nesting_level == 0, "Matching bracket not found in '%s'" % s
        ibefore = s.index(pattern)
        iafter = ibefore + len(sep) + pos + 2  # pylint:disable=W0631
        s = s[:ibefore] + cp2cond(s[ibefore:iafter]) + s[iafter:]
    return s

def prepend(prefix, words, string):
    """
    Prepend prefix to certain words in a string.
    
    >>> prepend("PREFIX", ["is", "test"], "This is a test: test_is")
    'This PREFIXis a PREFIXtest: test_is'
    """
    for w in words:
        # Use a r"raw string" because the word boundary \b 
        # means backspace in Python
        string = re.sub(r"\b%s\b" % w, prefix + w, string)
    return string

if __name__ == "__main__":
    import doctest
    doctest.testmod()
