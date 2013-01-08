"""Sensitivity analysis of action potential duration, bridging R and Python."""
# pylint: disable=C0301, R

# Numerics
import numpy as np
# Bridge to the R statistical software
import rpy2.rinterface as ri
from cgp.utils.rnumpy import r, py2ri
# Caching and on-demand recomputing
from joblib import Memory

# Wrapping ODE solver for CellML so it knows about variable names, etc.
from cgp.physmod.cellmlmodel import Cellmlmodel
# Virtual experiments
from cgp.phenotyping.attractor import AttractorMixin
from cgp.virtexp.elphys.clampable import Clampable
from cgp.virtexp.elphys.paceable import Paceable
# Utilities
from cgp.virtexp.elphys.paceable import ap_stats_array
from cgp.utils.unstruct import unstruct
from cgp.utils.failwith import failwithnanlikefirst

# Sensitivity analysis package for R
if __name__ == "__main__":  # allow nosetests to import without requiring this 
    r.library("sensitivity")
# Initialize caching
mem = Memory("/tmp/sensitivity_beeler")

class Model(Cellmlmodel, Clampable, Paceable, AttractorMixin):
    """Mix and match virtual experiments."""
    pass

# Wrap CellML model and adjust parameter names to conform with pacing protocol
m = Model(workspace="beeler_reuter_1977", rename=dict(
    p=dict(IstimPeriod="stim_period", IstimAmplitude="stim_amplitude", 
    IstimPulseDuration="stim_duration")), 
    reltol=1e-10, maxsteps=1e6, chunksize=100000)
m.pr.IstimStart = 0

# Will sample within plus/minus 50% of  all nonzero parameters
factors = [k for k in m.dtype.p.names if m.pr[k] != 0]

@mem.cache
@failwithnanlikefirst
def phenotypes(par=None):
    """Aggregate phenotypes for sensitivity analysis."""
    with m.autorestore(_p=par):
        m.eq(tmax=1e4, tol=1e-3)
        _t, _y, stats = m.ap()
    return ap_stats_array(stats)

phenotypes()  # initialize and cache default

def mat2par(mat):
    """Make parameter recarray from R matrix."""
    mat = np.copy(mat)
    par = np.tile(m.pr, len(mat))
    for i, factor in enumerate(factors):
        par[factor] = mat[:, i]
    return par

# pylint: disable=W0621
def scalar_pheno(field):
    """Make a function to return a named field of the phenotype array."""
    
    @ri.rternalize
    def fun(rmatrix):
        """Scalar function for use with R's sensitivity::morris()."""
        ph = np.concatenate([phenotypes(i) for i in mat2par(rmatrix)])
        return py2ri(ph[field])
    
    return fun

if __name__ == "__main__":
    # Sensitivity analysis
    baseline = unstruct(m.pr[factors])
    lower = 0.5 * baseline
    upper = 1.5 * baseline
    result = dict()
    for field in "appeak", "apd90", "ctpeak", "ctbase", "ctd90":
        r.set_seed(20120221)  # repeatable random sampling
        result[field] = r.morris(scalar_pheno(field), factors=factors, r=2, 
            design={"type": "oat", "levels": 10, "grid.jump": 5}, 
            binf=lower, bsup=upper)
    # Print and visualize results
    r.png("sensitivity.png", width=1024, height=768, pointsize=24)
    r.par(mfrow=(2, 3))  # multiple figures in two rows, three columns
    for k, v in result.items():
        print "===================================================="
        print k
        print v
        r.plot(v, log="y", main=k)
    r.dev_off()  # close image file
