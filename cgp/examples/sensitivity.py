"""Sensitivity analysis of action potential duration, bridging R and Python."""
# pylint: disable-all

import logging

import numpy as np
from matplotlib import pyplot as plt
from IPython.parallel import Client
import rpy2.rinterface as ri
from joblib import Memory

from cgp.utils.rnumpy import r, RRuntimeError, rcopy, py2ri
import cgp.virtexp.elphys.examples as ex
from cgp.phenotyping.attractor import AttractorMixin
from cgp.virtexp.elphys.paceable import ap_stats_array
from cgp.utils.unstruct import unstruct
from cgp.utils.rec2dict import rec2dict
from cgp.cvodeint.core import CvodeException
from cgp.utils.failwith import failwithnanlikefirst, failwithnan_asfor, failwith

# This requires that an ipcluster is already started, see
# http://ipython.org/ipython-doc/stable/parallel/parallel_task.html#parallel-function-decorator
# rc = Client()
# lview = rc.load_balanced_view()
try:
    r.library("sensitivity")
except RRuntimeError:
    r.install_packages("sensitivity", repos="http://cran.us.r-project.org")
    r.library("sensitivity")

class Model(ex.Bond, AttractorMixin):
    pass

logging.basicConfig()

r.set_seed(20120221)
m = Model(reltol=1e-10, maxsteps=1e6, chunksize=100000)
mem = Memory("/tmp/sensitivity")

factors = [k for k in m.dtype.p.names if m.pr[k] != 0]

@mem.cache
@failwithnanlikefirst
def phenotypes(par=None):
    with m.autorestore(_p=par):
        m.eq(tmax=1e7, tol=1e-3)
        t, y, stats = m.ap()
    return ap_stats_array(stats)

phenotypes()  # initialize and cache default

def mat2par(mat):
    """Make parameter recarray from R matrix."""
    mat = np.copy(mat)
    par = np.tile(m.pr, len(mat))
    for i, k in enumerate(factors):
        par[k] = mat[:, i]
    return par

@ri.rternalize
def converges(mat):
    np.save("mat2par.npy", mat2par(mat))
    ph = np.concatenate([phenotypes(i) for i in mat2par(mat)])
    np.save("ph.npy", ph)
    fail = [np.isnan(i.item()).all() for i in ph]
    result = np.logical_not(fail).astype(float)
    return py2ri(result)

@ri.rternalize
def apd90(mat):
    ph = np.concatenate([phenotypes(i) for i in mat2par(mat)])
    result = ph["apd90"]
    result[np.isnan(result)] = 0
    return py2ri(result)

@ri.rternalize
def appeak(mat):
    ph = np.concatenate([phenotypes(i) for i in mat2par(mat)])
    result = ph["appeak"]
    result[np.isnan(result)] = 0
    return py2ri(result)

#             mu   mu.star     sigma
#Cm   -0.4348831 0.6603604 0.9338907
#Vmyo  0.1264869 0.1264869 0.1657938
#             mu   mu.star     sigma
#Cm    0.6716927 0.6716927 0.3328038
#Vmyo -0.3135712 0.3135712 0.3033346

if __name__ == "__main__":
    binf = 0.5 * unstruct(m.pr[factors])
    bsup = 1.5 * unstruct(m.pr[factors])
    r.pdf("morrisplot.pdf")
    r.plot(r.morris(apd90, factors=factors, r=2, design={"type": "oat", "levels": 10, "grid.jump": 5}, binf=binf, bsup=bsup))
    r.dev_off()

#    saved = np.load("/tmp/saveme.npy")
#    ph = [phenotypes(i) for i in saved]
#    fail = [np.isnan(i).all() for i in ph]
#    result = np.logical_not(fail).astype(float)
