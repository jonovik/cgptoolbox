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
from cgp.rnumpy.rnumpy import r, py2ri, rwrap
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
@mem.cache()
def phenotypes(m, par=None):
    """
    Aggregate phenotypes for sensitivity analysis.
    
    >>> m = Model("11df840d0150d34c9716cd4cbdd164c8/bondarenko_szigeti_bett_kim_rasmusson_2004_apical")
    >>> print "Result:"; phenotypes(m)
    Result: ...
    rec.array([ (0.00013341746414141653, -82.4202, -82.42006658253585, 71.43, nan, 6.086027625230692, nan, nan, nan, 0.0, 0.115001, 0.115001, 0.0, nan, 0.0, 0.0, 0.0, 0.0)], 
          dtype=[('apamp', '<f8'), ('apbase', '<f8'), ('appeak', '<f8'), ('apttp', '<f8'), ('apdecayrate', '<f8'), ('apd25', '<f8'), ('apd50', '<f8'), ('apd75', '<f8'), ('apd90', '<f8'), ('ctamp', '<f8'), ('ctbase', '<f8'), ('ctpeak', '<f8'), ('ctttp', '<f8'), ('ctdecayrate', '<f8'), ('ctd25', '<f8'), ('ctd50', '<f8'), ('ctd75', '<f8'), ('ctd90', '<f8')])
    """
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
    >>> factors = ["Cm", "Vmyo"]  # must be list, not tuple
    >>> fun = scalar_pheno("apbase", m, factors)
    >>> # matrix = r.list(m.pr[factors])
    >>> # matrix
    >>> # r.do_call(r.str, r.list(m.pr[factors]))
    >>> from cgp.utils.unstruct import unstruct
    >>> r.do_call(fun, r.list(unstruct(m.pr[factors])))
    """
    
    @rwrap
    def fun(rmatrix):
        """Scalar function for use with R's sensitivity::morris()."""
        r.str(rmatrix)
        print repr(rmatrix)
        print rmatrix.shape
        ph = np.concatenate([phenotypes(m, i) for i in mat2par(rmatrix, m, factors)])
        return py2ri(ph[field])
    
    return fun

def _sensitivity(exposure, workspace, protocol, statistic, par=None, seed=None):
    """
    Sensitivity analysis.
    
    Callback from R.
    
    >>> r("fun <- function(func, x) func(x)")
    RClosure with name <return value from eval>:
    <R function>
    >>> r.fun(r.sum, range(5))
    array([10])
    (R-style, sealed)
    
    >>> _sensitivity("11df840d0150d34c9716cd4cbdd164c8", "bondarenko_szigeti_bett_kim_rasmusson_2004_apical", "protocol", "apd90", ("g_Na", "Nao"), seed=1)

    TODO: Make optional arguments of exposure, lower, upper, etc.
    TODO: Accept json dict of model_kwargs, morris_kwargs
    """
    m = Model(exposure + "/" + workspace, maxsteps=1e6, chunksize=1e5, reltol=1e-8)
    phenotypes(m)  # initialize and cache default
    par = [str(i) for i in par]  # Numpy cannot handle Unicode names
    if not par:
        # Default: sample within plus/minus 50% of all nonzero parameters
        par = [k for k in m.dtype.p.names if m.pr[k] != 0]
    baseline = unstruct(m.pr[par]).squeeze()
    lower = 0.5 * baseline
    upper = 1.5 * baseline
    if seed is not None:
        r.set_seed(seed)
    
    fun = scalar_pheno(m, statistic, par)
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
