"""Make Python callbacks from R work."""

import numpy as np
import rpy2.rinterface as ri

from cgp.rnumpy.rnumpy import r, py2ri

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
