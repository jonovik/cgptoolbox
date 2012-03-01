"""Make Python callbacks from R work."""

import numpy as np
import rpy2.rinterface as ri

from cgp.rnumpy.rnumpy import r, py2ri

r("funfun <- function(callback, x) callback(x)")

@ri.rternalize
def fun(x):
    return sum(x)

print r.funfun(fun, range(10))

@ri.rternalize
def y(rmatrix):
    r.str(rmatrix)
    return py2ri((10, 20, 30))

r.library("sensitivity")

print r.morris(y, factors=2, r=1, design={"type": "oat", "levels": 10, "grid.jump": 5})
