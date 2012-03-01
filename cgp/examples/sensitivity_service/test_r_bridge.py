"""Make Python callbacks from R work."""

import rpy2.rinterface as ri

from cgp.rnumpy.rnumpy import r

r("funfun <- function(callback, x) callback(x)")

@ri.rternalize
def fun(x):
    return sum(x)

print r.funfun(fun, range(10))