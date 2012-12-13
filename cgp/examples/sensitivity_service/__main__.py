# pylint: disable=R,C0301
"""
Web service for sensitivity analysis of CellML models.

Usage example:

wget http://localhost:8081 -O -

wget http://localhost:8081/sensitivity/ap/ctd50/beeler_reuter_1977?par=g_Na+E_Na -O -

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

# Web
import cgi
import bottle
from bottle import route, request

# Numerics
import numpy as np
# Bridge to the R statistical software
from cgp.utils.rnumpy import r, rwrap
# Caching and on-demand recomputing
from joblib import Memory

# Wrapping ODE solver for CellML so it knows about variable names, etc.
from cgp.physmod.cellmlmodel import Cellmlmodel
# Virtual experiments
from cgp.phenotyping.attractor import AttractorMixin
from cgp.virtexp.elphys.paceable import Paceable
# Utilities
from cgp.virtexp.elphys.paceable import ap_stats_array
from cgp.utils.unstruct import unstruct
from cgp.utils.failwith import failwithnanlikefirst

# Sensitivity analysis package for R
r.library("sensitivity")
# Initialize caching
mem = Memory("/tmp/sensitivity_service")

r.library("sensitivity")

# pylint: disable=W0105
'''def scalar_pheno(field):
    """Make a function to return a named field of the phenotype array."""
    
    @ri.rternalize
    def fun(rmatrix):
        """Scalar function for use with R's sensitivity::morris()."""
        ph = np.concatenate([phenotypes(i) for i in mat2par(rmatrix)])
        return py2ri(ph[field])
    
    return fun
'''

@route("/")
def index():
    """Usage instructions."""
    return "<pre>{}</pre>".format(cgi.escape(__doc__))

class Model(Cellmlmodel, Paceable, AttractorMixin):
    """CellML model wrapper with virtual experiments mixed in."""
    pass

@failwithnanlikefirst
@mem.cache()
def phenotypes(m, par=None):
    """
    Aggregate phenotypes for sensitivity analysis.
    
    Wrap CellML model and adjust parameter names to conform with pacing protocol
    
    >>> m = Model(workspace="beeler_reuter_1977", rename=dict(
    ...     p=dict(IstimPeriod="stim_period", IstimAmplitude="stim_amplitude", 
    ...     IstimPulseDuration="stim_duration")), 
    ...     reltol=1e-10, maxsteps=1e6, chunksize=100000)
    >>> m.pr.IstimStart = 0
    >>> print "Result:"; phenotypes(m)
    Result:...
    rec.array([ (115.799..., -83.857..., 31.942..., 2.287..., 0.01384..., 
    127.599..., 217.0..., 256.603..., 273.144..., 0.00612..., 0.000183..., 
    0.00630..., 92.373..., 0.0336..., 242.2..., 268.75..., 285.98..., 
    302.226...)], 
    dtype=[('apamp', '<f8'), ('apbase', '<f8'), ('appeak', '<f8'), 
    ('apttp', '<f8'), ('apdecayrate', '<f8'), ('apd25', '<f8'), 
    ('apd50', '<f8'), ('apd75', '<f8'), ('apd90', '<f8'), ('ctamp', '<f8'), 
    ('ctbase', '<f8'), ('ctpeak', '<f8'), ('ctttp', '<f8'), 
    ('ctdecayrate', '<f8'), ('ctd25', '<f8'), ('ctd50', '<f8'), 
    ('ctd75', '<f8'), ('ctd90', '<f8')])
    """
    with m.autorestore(_p=par):
        m.eq(tmax=1e4, tol=1e-3)
        _t, _y, stats = m.ap()
    return ap_stats_array(stats)

def mat2par(mat, m, factors):
    """
    Make parameter recarray from R matrix.
    
    >>> m = Model(workspace="bondarenko_szigeti_bett_kim_rasmusson_2004_apical",
    ...     exposure="11df840d0150d34c9716cd4cbdd164c8")
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
    Make an @rwrap'ed function to return a named field of the phenotype array.
    
    The resulting function can only be called from R, not Python. 
    The input should be a matrix; each row will be passed to phenotypes().
    
    >>> m = Model(workspace="bondarenko_szigeti_bett_kim_rasmusson_2004_apical",
    ...     exposure="11df840d0150d34c9716cd4cbdd164c8")
    >>> factors = ["Cm", "Vmyo"]  # must be list, not tuple
    >>> caller = r("function(callback, x) callback(x)")
    >>> callback = scalar_pheno("apbase", m, factors)
    >>> input = np.reshape(m.pr[factors].item(), (1, -1))
    >>> print "apbase:", caller(callback, input)
    apbase:...[-82.4202]
    """
    
    @rwrap
    def fun(rmatrix):
        """Scalar function for use with R's sensitivity::morris()."""
        par = mat2par(rmatrix, m, factors)
        ph = np.concatenate([phenotypes(m, p) for p in par])
        return ph[field]
    
    return fun

def do_sensitivity(exposure, workspace, protocol, statistic, par=None, 
    seed=None, model=None, changeset=None, variant=None):
    """
    Sensitivity analysis.
    
    Callback from R.
    
    >>> r("fun <- function(func, x) func(x)")
    RClosure with name <return value from eval>:
    <R function>
    >>> r.fun(r.sum, range(5))
    array([10])
    (R-style, sealed)
    
    >>> m = Model(workspace="beeler_reuter_1977", rename=dict(
    ...     p=dict(IstimPeriod="stim_period", IstimAmplitude="stim_amplitude", 
    ...     IstimPulseDuration="stim_duration", IstimStart="stim_start")), 
    ...     reltol=1e-10, maxsteps=1e6, chunksize=100000)
    >>> m.pr.IstimStart = 0
    >>> print "Result:", do_sensitivity("", "", "protocol", "apbase", ("C", "g_Na"), seed=1, model=m)
    Result:...
    Model runs: 6 
                  mu    mu.star      sigma
    C     0.03672701 0.03672701 0.05130616
    g_Na -0.41200934 0.41200934 0.04219007

    TODO: Make optional arguments of exposure, lower, upper, etc.
    TODO: Accept json dict of model_kwargs, morris_kwargs
    """
    if model is None:
        m = Model(workspace, exposure, changeset, variant, 
            maxsteps=1e6, chunksize=1e5, reltol=1e-8)
        m.pr.stim_start = 0
    else:
        m = model
    phenotypes(m)  # initialize and cache default
    factors = [str(i) for i in par]  # Numpy cannot handle Unicode names
    if not factors:
        # Default: sample within plus/minus 50% of all nonzero parameters
        factors = [k for k in m.dtype.p.names if m.pr[k] != 0]
    baseline = unstruct(m.pr[factors]).squeeze()
    lower = 0.5 * baseline
    upper = 1.5 * baseline
    if seed is not None:
        r.set_seed(seed)
    fun = scalar_pheno(statistic, m, factors)
    design = {"type": "oat", "levels": 10, "grid.jump": 5}
    return r.morris(fun, factors=factors, r=2, design=design, 
        binf=lower, bsup=upper)

@route("/sensitivity/<exposure>/<workspace>/<protocol>/<statistic>")
def sensitivity(exposure, workspace, protocol, statistic):
    """Run sensitivity analysis based on parsed query arguments."""
    par = request.params.par.split()  # space-delimited string
    # par = json.loads(request.params.par)  # JSON or Python
    # par = request.params.getall("par")  # HTML multiple SELECT
    # seed = request.params.seed  # Specify random seed for repeatability
    m = Model(workspace="beeler_reuter_1977", rename=dict(
        p=dict(IstimPeriod="stim_period", IstimAmplitude="stim_amplitude", 
        IstimPulseDuration="stim_duration", IstimStart="stim_start")), 
        reltol=1e-10, maxsteps=1e6, chunksize=100000)
    m.pr.IstimStart = 0
    return "<pre>%s</pre>" % do_sensitivity(exposure, workspace, protocol, 
        statistic, par=par, seed=None, model=m)

if __name__ == "__main__":
    bottle.run(host='localhost', port=8080, debug=True, reloader=True)
