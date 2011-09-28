"""
Figure: Separation of concerns aids reusability of virtual experiments.

The figure uses multiple subplots to illustrate alternatives for each 
component. How to build interactive HTML with hyperlinks: 
http://www.dalkescientific.com/writings/diary/archive/2005/04/24/interactive_html.html

SVG with hyperlinks:
http://matplotlib.sourceforge.net/examples/pylab_examples/hyperlinks.html

Todo: Separate function for each panel. They need separate limits, etc., so 
is easier to define as function than pass as arguments.
Todo: Add illustration for models.
"""

from matplotlib import pyplot as plt
import os
from joblib import Memory
from ap_cvode import Bond
from protocols import catrec, Clampable, markovplot
from splom import spij

### Options, model and protocols

cell = Clampable.mixin(Bond)
protocols = cell.bond_protocols()
# Pacing parameters
nprepace = 10
npace = 2
# P1-P2 double-pulse protocol to study voltage-dependence of ion channel
p1p2 = protocols[3]
# Variable-gap protocol to study the rate of recovery from inactivation
vargap = protocols[7]
# Lineplot options
lineplotopt = dict(linestyle="-", color="k", linewidth=2)
markovopt = dict(model=cell, plotpy=True, plotr=False, newfig=False)


mem = Memory(os.path.join(os.environ["TEMP"], "fig_virtual_experiments"))

@mem.cache
def vecvclamp(*args, **kwargs):
    return cell.vecvclamp(*args, **kwargs)

### Regular pacing

# Prepacing
for t, y, stats in cell.aps(n=nprepace):
    pass

# Two paces
t, y, stats = catrec(*cell.aps(n=npace), globalize_time=False)
dy, a = cell.rates_and_algebraic(t, y)

def lineplot(x, y, title, ylabel):
    def result():
        plt.plot(x, y, **lineplotopt)
        plt.title(title)
        plt.ylabel(ylabel)
    return result

exp_pace = lineplot(t, a.i_stim, "Pacing", "Stimulus current")
vis_ap = lineplot(t, y.V, "Action potential", "Voltage")
vis_ct = lineplot(t, y.Cai, "Calcium transient", "[Ca]")


### Double-pulse protocol

L = vecvclamp(p1p2.protocol)
proto, traj = L[3]
t, y, dy, a = catrec(*traj)
exp_p1p2 = lineplot(t, y["V"], "Double-pulse clamping", "Voltage")

def vis_p1p2():
    markovplot(t, y, comp="fast_sodium", **markovopt)


### Variable-gap protocol

L = vecvclamp(vargap.protocol)  # list of (protocol, trajectories)
proto, traj = L[3]
t, y, dy, a = catrec(*traj)
exp_vargap = lineplot(t, y["V"], "Variable-gap clamping", "Voltage")

def vis_vargap():
    markovplot(t, y, comp="L_type", **markovopt)


### List of models

def model(name):
    def result():
        plt.title(name)
    return result

mod = [model(i) 
    for i in "Hodgkin-Huxley FitzHugh-Nagumo Luo-Rudy Bondarenko".split()]
exp = [exp_pace, exp_p1p2, exp_vargap]
vis = [vis_ap, vis_ct, vis_p1p2, vis_vargap]

panels = mod, exp, vis

m = max([len(i) for i in panels])
n = len(panels)

plt.figure()
for j, panelj in enumerate(panels):
    for i, panelij in enumerate(panelj):
        spij(m, n, i, j)
        panelij()

plt.show()
