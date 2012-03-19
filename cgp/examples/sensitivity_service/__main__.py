"""
Web service for sensitivity analysis of CellML models.

Usage example:

http://localhost:8080/sensitivity/11df840d0150d34c9716cd4cbdd164c8/bondarenko_szigeti_bett_kim_rasmusson_2004_apical/protocol/statistic?par=g_Na+Nao

In general:

http://localhost:8080/sensitivity/<exposure>/<workspace>/<protocol>/<statistic>?par=p1+p2

will analyse the sensitivity with respect to parameters p1 and p2
of a <statistic> that characterizes the outcome of an experimental <protocol>
applied to a CellML model identified by <exposure>/<workspace> at cellml.org.

Parameters can be passed by GET (shown above) or POST (suitable for long 
parameter lists).

Valid formats for the query string for the parameter list are:
par=p1+p2          space-delimited
TODO: further formats
par=p1&par=p2      "SELECT multiple" HTML query
par=["p1", "p2"]   JSON, Python

TODO:

1. Dropdown list of all CellML models.
1a. Restrict list to CellML models that have V.
2. Radio buttons for all parameters to select targets for sensitivity analysis.
3. Autogenerate limits based on percentage change. 
4. Set lower and upper limits manually.
5. Present results as ASCII (melt) or JSON.
6. Allow tweaking of options to Morris or virtual experiment.
7. AJAX instead of re-submitting on every input.
"""
# pylint: disable=C0301, R

# Web
import cgi
import json
import bottle
from bottle import route, run, view, request

# Numerics
import numpy as np
# Bridge to the R statistical software
import rpy2.rinterface as ri
from cgp.rnumpy.rnumpy import r, py2ri
# Caching and on-demand recomputing
from joblib import Memory

# Wrapping ODE solver for CellML so it knows about variable names, etc.
from cgp.physmod.cellmlmodel import Cellmlmodel, get_latest_exposure
# Virtual experiments
from cgp.phenotyping.attractor import AttractorMixin
from cgp.virtexp.elphys.clampable import Clampable
from cgp.virtexp.elphys.paceable import Paceable
# Utilities
from cgp.virtexp.elphys.paceable import ap_stats_array
from cgp.utils.unstruct import unstruct
from cgp.utils.failwith import failwithnanlikefirst

# Sensitivity analysis package for R
r.library("sensitivity")
# Initialize caching
mem = Memory("/tmp/sensitivity_service")

@route("/")
def index():
    """Usage instructions."""
    return "<pre>{}</pre>".format(cgi.escape(__doc__))

class Model(Cellmlmodel, Paceable, AttractorMixin):
    """CellML model wrapper with virtual experiments mixed in."""
    pass

mem.clear()

@failwithnanlikefirst
@mem.cache(ignore=["m"])
def phenotypes(m, modelname, par=None):
    """Aggregate phenotypes for sensitivity analysis."""
    with m.autorestore(_p=par):
        m.eq(tmax=1e4, tol=1e-3)
        _t, _y, stats = m.ap()
    return ap_stats_array(stats)

def mat2par(mat, m, factors):
    """
    Make parameter recarray from R matrix.
    
    >>> m = Model("11df840d0150d34c9716cd4cbdd164c8/bondarenko_szigeti_bett_kim_rasmusson_2004_apical")
    >>> mat2par(r.matrix(range(4), ncol=2), m, ["Cm", "Vmyo"])
    rec.array([ (0.0, 2.0, 1.2e-07, ... 
                (1.0, 3.0, 1.2e-07, ...
                dtype=[('Cm', '<f8'), ('Vmyo', '<f8'), ('VJSR', '<f8'), ...
    """
    mat = np.copy(mat)
    par = np.tile(m.pr, len(mat))
    for i, k in enumerate(factors):
        par[k] = mat[:, i]
    return par

def scalar_pheno(field, m, factors):
    """
    Make a function to return a named field of the phenotype array.
    
    >>> m = Model("11df840d0150d34c9716cd4cbdd164c8/bondarenko_szigeti_bett_kim_rasmusson_2004_apical")
    >>> factors = "Cm", "Vmyo"
    >>> phenotypes(m, m.name)
    >>> fun = scalar_pheno("apd90", m, factors)
    >>> rmatrix = r.matrix(m.pr.item(), nrow=1)
    >>> type(rmatrix)
    >>> mat2par(rmatrix, m, factors)
    >>> fun(r.matrix(m.pr.item(), nrow=1))
    """
    
    @ri.rternalize
    def fun(rmatrix):
        """Scalar function for use with R's sensitivity::morris()."""
        ph = np.concatenate([phenotypes(m, m.name, i) for i in mat2par(rmatrix, m, factors)])
        return py2ri(ph[field])
    
    return fun

def _sensitivity(exposure, workspace, protocol, statistic, par=None, seed=None):
    """
    Sensitivity analysis.
    
    >>> _sensitivity("11df840d0150d34c9716cd4cbdd164c8", "bondarenko_szigeti_bett_kim_rasmusson_2004_apical", "protocol", "apd90", ("g_Na", "Nao"), seed=1)

    TODO: Make optional arguments of exposure, lower, upper, etc.
    TODO: Accept json dict of model_kwargs, morris_kwargs
    """
    m = Model(exposure + "/" + workspace, maxsteps=1e6, chunksize=1e5, reltol=1e-8)
    phenotypes(m, m.name)  # initialize and cache default
    par = [str(i) for i in par]  # Numpy cannot handle Unicode names
    if not par:
        # Default: sample within plus/minus 50% of all nonzero parameters
        par = [k for k in m.dtype.p.names if m.pr[k] != 0]
    baseline = unstruct(m.pr[par]).squeeze()
    lower = 0.5 * baseline
    upper = 1.5 * baseline
    if seed is not None:
        r.set_seed(seed)
    
    r("fun <- function(func, x) func(x)")
    raise Exception(str(r.fun(r.sum, baseline)))

    
    fun = scalar_pheno(m, statistic)
    r.as_vector(baseline)
    return r.morris(fun, factors=par, r=2, design={"type": "oat", "levels": 10, "grid.jump": 5}, binf=lower, bsup=upper)

#@route("/sensitivity/<exposure>/<workspace>/<protocol>/<statistic>")
def sensitivity(exposure, workspace, protocol, statistic):
    """Run sensitivity analysis based on parsed query arguments."""
    par = request.params.par.split()  # space-delimited string
    # par = json.loads(request.params.par)  # JSON or Python
    # par = request.params.getall("par")  # HTML multiple SELECT
    seed = request.params.seed
    return _sensitivity(exposure, workspace, protocol, statistic, par=("g_Na", "Nao"), seed=None, **kwargs)

if __name__ == "__main__":
    bottle.run(host='localhost', port=8080, debug=True, reloader=True)
