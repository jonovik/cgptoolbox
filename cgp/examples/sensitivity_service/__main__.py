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

import bottle
from bottle import route, run, view
from cgp.physmod.cellmlmodel import get_all_workspaces
from joblib.memory import Memory
from cgp.utils.rec2dict import rec2dict

mem = Memory("/tmp/sensitivity")

get_all_workspaces = mem.cache(get_all_workspaces)

@route("/sensitivity")
@view("sensitivity")
def sensitivity():
    return dict(model=get_all_workspaces())

# webbrowser.open("http://localhost:8080/sensitivity")
bottle.run(host='localhost', port=8080, debug=True, reloader=True)
