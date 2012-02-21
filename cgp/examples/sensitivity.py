"""Sensitivity analysis of action potential duration, bridging R and Python."""

import numpy as np
from matplotlib import pyplot as plt
from IPython.parallel import Client
import rpy2.rinterface as ri
from joblib import Memory

from cgp.rnumpy.rnumpy import r, RRuntimeError, rcopy, py2ri
import cgp.virtexp.elphys.examples as ex
from cgp.phenotyping.attractor import AttractorMixin
from cgp.virtexp.elphys.paceable import ap_stats_array
from cgp.utils.unstruct import unstruct

# This requires that an ipcluster is already started, see
# http://ipython.org/ipython-doc/stable/parallel/parallel_task.html#parallel-function-decorator
# rc = Client()
# lview = rc.load_balanced_view()
try:
    r.library("sensitivity")
except RRuntimeError:
    r.install_packages("sensitivity")
    r.library("sensitivity")

class Model(ex.Bond, AttractorMixin):
    pass

r.set_seed(20120221)
m = Model(reltol=1e-10, maxsteps=1e6, chunksize=100000)
mem = Memory("/tmp/sensitivity")

#             mu   mu.star     sigma
#Cm   -0.4348831 0.6603604 0.9338907
#Vmyo  0.1264869 0.1264869 0.1657938

factors = ["Cm", "Vmyo"]

@ri.rternalize
# @lview.parallel
@mem.cache
def apd90(par):
    """Input is a matrix (with colnames if factors is string)."""
    par = np.copy(par)  # Convert from low-level R object
    result = []
    for i in par:
        kwargs = dict(zip(factors, par[0]))
        with m.autorestore(**kwargs):
            m.eq(tmax=1e6, tol=1e-4)
    #        try:
    #            m.eq(tmax=1e6, tol=1e-4)
    #        except Exception, exc:
    #            t, y, flag = exc.result
    #            print str(exc)
    #            plt.plot(t, y)
    #            plt.show()
            t, y, stats = m.ap()
        result.append(ap_stats_array(stats).apd90)
    return py2ri(result)  # Convert to low-level R object

print r.morris(apd90, factors=factors, r=2, design={"type": "oat", "levels": 10, "grid.jump": 5}, binf=0.99 * unstruct(m.pr), bsup=1.01 * unstruct(m.pr))

#@ri.rternalize
#def fun(v):
#    x, y = np.transpose(v)
#    z = 2 * y * np.sin(x) * np.cos(x) * np.arctan(y)
#    return py2ri(z)
#
# print r.morris(fun, factors=2, r=2, design={"type": "oat", "levels": 10, "grid.jump": 5})
