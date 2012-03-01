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
@mem.cache()
def phenotypes(m, par=None):
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
    >>> print "Result:", repr(phenotypes(m))
    Result: ...
    rec.array([ (0.00013341746414141653, -82.4202, -82.42006658253585, 71.43, nan, 6.086027625230692, nan, nan, nan, 0.0, 0.115001, 0.115001, 0.0, nan, 0.0, 0.0, 0.0, 0.0)], 
          dtype=[('apamp', '<f8'), ('apbase', '<f8'), ('appeak', '<f8'), ('apttp', '<f8'), ('apdecayrate', '<f8'), ('apd25', '<f8'), ('apd50', '<f8'), ('apd75', '<f8'), ('apd90', '<f8'), ('ctamp', '<f8'), ('ctbase', '<f8'), ('ctpeak', '<f8'), ('ctttp', '<f8'), ('ctdecayrate', '<f8'), ('ctd25', '<f8'), ('ctd50', '<f8'), ('ctd75', '<f8'), ('ctd90', '<f8')])
    >>> fun = scalar_pheno("apd90", m, factors)
    >>> rmatrix = r.matrix(m.pr.item(), nrow=1)
    >>> type(rmatrix)
    <class 'cgp.rnumpy.rnumpy.RArray'>
    >>> mat2par(rmatrix, m, factors)
    rec.array([ (1.0, 2.584e-05, 1.2e-07, 2.098e-06, 1.485e-09, 0.0001534, 5400.0, 140000.0, 1800.0, 8.314, 298.0, 96.5, 20.0, 100000.0, 71.43, 0.5, -80.0, 50.0, 15000.0, 0.238, 800.0, 0.00237, 3.2e-05, 0.0327, 0.0196, 4.5, 20.0, 1.74e-05, 8.0, 0.45, 0.5, 70.0, 140.0, 7.0, 0.006075, 0.07125, 0.00405, 0.965, 0.009, 0.0008, 3.0, 4.0, 63.0, 0.1729, 0.0005, 0.23324, 20.0, 1.0, 0.5, 292.8, 87500.0, 1380.0, 0.1, 0.35, 0.000367, 13.0, 0.0026, 0.4067, 0.0, 0.00575, 0.16, 0.05, 0.078, 0.036778, 0.023761, 0.88, 21000.0, 1500.0, 10.0, -40.0, 10.0, 1.0009103049457284, 0.0)], 
          dtype=[('Cm', '<f8'), ('Vmyo', '<f8'), ('VJSR', '<f8'), ('VNSR', '<f8'), ('Vss', '<f8'), ('Acap', '<f8'), ('Ko', '<f8'), ('Nao', '<f8'), ('Cao', '<f8'), ('R', '<f8'), ('T', '<f8'), ('F', '<f8'), ('stim_start', '<f8'), ('stim_end', '<f8'), ('stim_period', '<f8'), ('stim_duration', '<f8'), ('stim_amplitude', '<f8'), ('CMDN_tot', '<f8'), ('CSQN_tot', '<f8'), ('Km_CMDN', '<f8'), ('Km_CSQN', '<f8'), ('k_plus_htrpn', '<f8'), ('k_minus_htrpn', '<f8'), ('k_plus_ltrpn', '<f8'), ('k_minus_ltrpn', '<f8'), ('v1', '<f8'), ('tau_tr', '<f8'), ('v2', '<f8'), ('tau_xfer', '<f8'), ('v3', '<f8'), ('Km_up', '<f8'), ('LTRPN_tot', '<f8'), ('HTRPN_tot', '<f8'), ('i_CaL_max', '<f8'), ('k_plus_a', '<f8'), ('k_minus_a', '<f8'), ('k_plus_b', '<f8'), ('k_minus_b', '<f8'), ('k_plus_c', '<f8'), ('k_minus_c', '<f8'), ('m', '<f8'), ('n', '<f8'), ('E_CaL', '<f8'), ('g_CaL', '<f8'), ('Kpcb', '<f8'), ('Kpc_max', '<f8'), ('Kpc_half', '<f8'), ('i_pCa_max', '<f8'), ('Km_pCa', '<f8'), ('k_NaCa', '<f8'), ('K_mNa', '<f8'), ('K_mCa', '<f8'), ('k_sat', '<f8'), ('eta', '<f8'), ('g_Cab', '<f8'), ('g_Na', '<f8'), ('g_Nab', '<f8'), ('g_Kto_f', '<f8'), ('g_Kto_s', '<f8'), ('g_Ks', '<f8'), ('g_Kur', '<f8'), ('g_Kss', '<f8'), ('g_Kr', '<f8'), ('kb', '<f8'), ('kf', '<f8'), ('i_NaK_max', '<f8'), ('Km_Nai', '<f8'), ('Km_Ko', '<f8'), ('g_ClCa', '<f8'), ('E_Cl', '<f8'), ('Km_Cl', '<f8'), ('sigma', '<f8'), ('empty__72', '<f8')])
    >>> fun(r.matrix(m.pr.item(), nrow=1))
    """
    
    @ri.rternalize
    def fun(rmatrix):
        """Scalar function for use with R's sensitivity::morris()."""
        r.str(rmatrix)
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
