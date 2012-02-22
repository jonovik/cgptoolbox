"""Sensitivity analysis of action potential duration, bridging R and Python."""
# pylint: disable=C0301, R

import logging

import numpy as np
#from matplotlib import pyplot as plt
#from IPython.parallel import Client
import rpy2.rinterface as ri
from joblib import Memory

from cgp.physmod.cellmlmodel import Cellmlmodel
from cgp.phenotyping.attractor import AttractorMixin
from cgp.virtexp.elphys.clampable import Clampable
from cgp.virtexp.elphys.paceable import ap_stats_array, Paceable
from cgp.rnumpy.rnumpy import r, RRuntimeError, py2ri, rcopy
from cgp.utils.unstruct import unstruct
from cgp.utils.failwith import failwithnanlikefirst

# This requires that an ipcluster is already started, see
# http://ipython.org/ipython-doc/stable/parallel/parallel_task.html#parallel-function-decorator
# rc = Client()
# lview = rc.load_balanced_view()
try:
    r.library("sensitivity")
except RRuntimeError:
    r.install_packages("sensitivity")
    r.library("sensitivity")

logging.basicConfig()
mem = Memory("/tmp/sensitivity_beeler")

class Model(Cellmlmodel, Clampable, Paceable, AttractorMixin):
    """Mix and match virtual experiments."""
    pass

m = Model("/beeler_reuter_1977", rename=dict(
    p=dict(IstimPeriod="stim_period", IstimAmplitude="stim_amplitude", 
    IstimPulseDuration="stim_duration")), 
    reltol=1e-10, maxsteps=1e6, chunksize=100000)
m.pr.IstimStart = 0

r.set_seed(20120221)

factors = [k for k in m.dtype.p.names if m.pr[k] != 0]

#@failwithnanlikefirst
def _phenotypes(par=None):
    """Aggregate phenotypes for sensitivity analysis."""
    with m.autorestore(_p=par):
        m.eq(tmax=1e4, tol=1e-3)
        _t, _y, stats = m.ap()
    return ap_stats_array(stats)

phenotypes = mem.cache(_phenotypes)  # avoid name conflict in joblib

phenotypes()  # initialize and cache default

def mat2par(mat):
    """Make parameter recarray from R matrix."""
    mat = np.copy(mat)
    par = np.tile(m.pr, len(mat))
    for i, k in enumerate(factors):
        par[k] = mat[:, i]
    return par

@ri.rternalize
def apd50(mat):
    """Scalar function for use with R's sensitivity::morris()."""
    ph = np.concatenate([phenotypes(i) for i in mat2par(mat)])
    # np.save("sbmat.npy", mat)
    # np.save("sbpar.npy", mat2par(mat))
    # np.save("sbph.npy", ph)
    result = ph["apd50"]
    result[np.isnan(result)] = 0
    return py2ri(result)

if __name__ == "__main__":
    binf = 0.5 * unstruct(m.pr[factors]).squeeze()
    bsup = 1.5 * unstruct(m.pr[factors]).squeeze()
    morr = r.morris(apd50, factors=factors, r=2, design={"type": "oat", "levels": 10, "grid.jump": 5}, binf=binf, bsup=bsup)
    print morr
#    r["morr"] = morr
#    r.save("morr", file="morr.RData")
#    r.str(morr)
#    r.tell(morr)
