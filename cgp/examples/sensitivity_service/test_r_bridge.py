"""Make Python callbacks from R work."""

import numpy as np
import rpy2.rinterface as ri

from cgp.utils.rnumpy import r, py2ri
from cgp.virtexp.elphys.examples import Bond

r("funfun <- function(callback, x) callback(x)")

@ri.rternalize
def fun(x):
    return sum(x)

print r.funfun(fun, range(10))  # Simple callback

@ri.rternalize  # Required for use as callback from R
def y(rmatrix):
    r.str(rmatrix)
    x = np.copy(rmatrix)  # For easier handling in Python
    result = x.sum(axis=1)
    return py2ri(result)  # Required for return to R

r.library("sensitivity")

# morris parameters:
# factors >= 2 is required (presumably to estimate mu)
# r >= 2 is required to estimate sigma

print r.morris(y, factors=2, r=2, design={"type": "oat", "levels": 10, "grid.jump": 5})

m = Bond()

@ri.rternalize
def pheno(rmatrix):
    result = []
    arr = np.copy(rmatrix)
    for g_Na, Nao in arr:
        with m.autorestore(g_Na=g_Na, Nao=Nao):
            t, y, stats = m.ap()
        result.append(stats["peak"])
    return py2ri(result)

# Works fine:
print r.funfun(pheno, [(m.pr.g_Na, m.pr.Nao) for i in range(2)])

# raises ValueError: All parameters must be of type Sexp_Type,or Python int/long, float, bool, or None
try:
    pheno([(m.pr.g_Na, m.pr.Nao) for i in range(10)])
except ValueError, exc:
    print exc

def unwrapped(arr):
    return arr.sum(axis=1)

def wrap(func):
    
    def wrapper(rmatrix):
        arr = np.copy(rmatrix)
        result = func(arr)
        return py2ri(result)
    
    return wrapper

print r.funfun(ri.rternalize(wrap(unwrapped)), np.arange(10).reshape(5, 2))

@ri.rternalize
@wrap
def decorated(arr):
    return arr.sum(axis=0)

print r.funfun(decorated, np.arange(10).reshape(5, 2))

def twowrap(func):
    
    @ri.rternalize
    def wrapper(rmatrix):
        arr = np.copy(rmatrix)
        result = func(arr)
        return py2ri(result)
    
    return wrapper

@twowrap
def twowrapped(arr):
    return arr.sum(axis=1)

print r.funfun(twowrapped, np.arange(10).reshape(5, 2))
