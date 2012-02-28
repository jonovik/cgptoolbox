"""
Web service for sensitivity analysis of CellML models.



1. Dropdown list of all CellML models.
1a. Restrict list to CellML models that have V.
2. Radio buttons for all parameters to select targets for sensitivity analysis.
3. Autogenerate limits based on percentage change. 
4. Set lower and upper limits manually.
5. Present results as ASCII (melt) or JSON.
6. Allow tweaking of options to Morris or virtual experiment.
7. AJAX instead of re-submitting on every input.
"""

import webbrowser
import urllib

import bottle
from bottle import route, run, view
from cgp.physmod.cellmlmodel import get_all_workspaces, Cellmlmodel
from joblib.memory import Memory
from cgp.utils.rec2dict import rec2dict

mem = Memory("/tmp/sensitivity")

get_all_workspaces = mem.cache(get_all_workspaces)

@route("/")
def index():
    return "<pre>{}</pre>".format(__doc__)

@route("/sensitivity")
@view("sensitivity")
def sensitivity():
    # TODO: use workspace as identifier, get_latest_exposure
    d = dict(workspaces=get_all_workspaces())
    d["path"], _ = urllib.splitquery(bottle.request.url)
    d["query"] = bottle.request.query
    try:
        d["model"] = Cellmlmodel("/fitzhugh_1961")
        # d["model"] = Cellmlmodel(bottle.request.query.workspace)
    except ZeroDivisionError:
        d["model"] = None
    return d

# webbrowser.open("http://localhost:8080/sensitivity")
bottle.run(host='localhost', port=8080, debug=True, reloader=True)
