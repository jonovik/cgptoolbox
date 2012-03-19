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

import cgi
import json

import bottle
from bottle import route, run, view, request
from ast import literal_eval
from cgp.physmod.cellmlmodel import Cellmlmodel
from cgp.virtexp.elphys.paceable import Paceable
from cgp.utils.unstruct import unstruct

# Numerics
import numpy as np
# Bridge to the R statistical software
import rpy2.rinterface as ri
from cgp.rnumpy.rnumpy import r, py2ri

r.library("sensitivity")

def scalar_pheno(field):
    """Make a function to return a named field of the phenotype array."""
    
    @ri.rternalize
    def fun(rmatrix):
        """Scalar function for use with R's sensitivity::morris()."""
        ph = np.concatenate([phenotypes(i) for i in mat2par(rmatrix)])
        return py2ri(ph[field])
    
    return fun


@route("/")
def index():
    """Usage instructions."""
    return "<pre>{}</pre>".format(cgi.escape(__doc__))

class Model(Cellmlmodel, Paceable):
    """CellML model wrapper with phenotyping mixed in."""
    pass

@route
def sensitivity(protocol, statistic, workspace, exposure="", changeset="", variant=""):
    """
    Sensitivity analysis.
    
    TODO: Make exposure optional.
    """
    par = request.params.par.split()  # space-delimited string
    # par = json.loads(request.params.par)  # JSON or Python
    # par = request.params.getall("par")  # HTML multiple SELECT
    par = [str(i) for i in par]  # Numpy cannot handle Unicode names
    try:
        m = Model(workspace, exposure, changeset, variant, maxsteps=1e6, chunksize=1e5, reltol=1e-8)
    except IOError, exc:
        return "<pre>{}</pre>".format(exc)
    baseline = unstruct(m.pr[par]).squeeze()
    lower = 0.5 * baseline
    upper = 1.5 * baseline
    # return dict(lower=lower.tolist(), upper=upper.tolist())
    r.set_seed(request.params.seed or 1)
    return r.morris(scalar_pheno(statistic), factors=par, r=2, design={"type": "oat", "levels": 10, "grid.jump": 5}, binf=lower, bsup=upper)

#@route("/sensitivity")
#@view("sensitivity")
#def sensitivity():
#    # TODO: use workspace as identifier, get_latest_exposure
#    d = dict(workspaces=get_all_workspaces())
#    d["path"], _ = urllib.splitquery(bottle.request.url)
#    d["query"] = bottle.request.query
#    try:
#        d["model"] = Cellmlmodel("/fitzhugh_1961")
#        # d["model"] = Cellmlmodel(bottle.request.query.workspace)
#    except ZeroDivisionError:
#        d["model"] = None
#    return d

if __name__ == "__main__":
    bottle.run(debug=True, reloader=True, port=8081)
