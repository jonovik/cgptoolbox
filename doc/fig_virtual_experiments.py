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

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import (plot, ylabel, title, 
    tick_params, xlim, axis, box, setp, getp)
import os
from joblib import Memory
from cgp.virtexp.elphys.examples import Bond
from cgp.virtexp.elphys.clampable import catrec, markovplot, listify, Bond_protocol
from cgp.utils.splom import spij

### Options, model and protocols

cell = Bond()
protocols = cell.bond_protocols()
# Pacing parameters
nprepace = 10
npace = 4
# P1-P2 double-pulse protocol to study voltage-dependence of ion channel
varnames, protocol, limits, url = listify(protocols[3])
protocol[1][1] = protocol[1][1][::3]
p1p2 = Bond_protocol(varnames, protocol, limits, url)
# Variable-gap protocol to study the rate of recovery from inactivation
varnames, protocol, limits, url = listify(protocols[7])
protocol[2][0] = protocol[2][0][6::3]  # don't need so many gap lengths
protocol[2][1] = protocol[2][1][1]  # don't vary gap voltage
protocol[3][0] = 300  # make shortest run last until the longest gap is done
vargap = Bond_protocol(varnames, protocol, limits, url)

# Lineplot options
lineplotopt = dict(linestyle="-", color="k", linewidth=2)
markovopt = dict(model=cell, plotpy=True, plotr=False, newfig=False, loc="upper left")


mem = Memory(os.path.join(os.environ.get("TEMP", "/tmp"), "fig_virtual_experiments"))

@mem.cache
def vecvclamp(*args, **kwargs):
    return cell.vecvclamp(*args, **kwargs)

### Regular pacing

@mem.cache
def pacing_output():
    # Prepacing
    for _t, _y, _stats in cell.aps(n=nprepace):
        pass
    # A few paces
    t, y, _stats = catrec(*cell.aps(n=npace), globalize_time=False)
    dy, a = cell.rates_and_algebraic(t, y)
    return t, y, dy, a

t, y, dy, a = pacing_output()
pacelim = np.r_[(cell.pr.stim_period / 2), (t[-1] - cell.pr.stim_period / 2)]


def tweak():
    xmin, xmax, ymin, ymax = axis()
    xpad = 0.1 * (xmax - xmin)
    ypad = 0.1 * (ymax - ymin)
    axis([xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad])
    tick_params(length=0)
    box("off")

def tweak_legend():
    plt.gca().get_legend().texts[0].set_size("xx-small")

# Hack: Use default parameters as hack to store current value of a before it 
# gets redefined below.
# http://stackoverflow.com/questions/233673/lexical-closures-in-python#235764

def exp_pace(t=t, a=a):
    plot(t, a["i_stim"], **lineplotopt)
    title("Pacing")
    ylabel("Stimulus current")
    tweak()
    xlim(pacelim)

def vis_ap(t=t, y=y):
    plot(t, y["V"], **lineplotopt)
    title("Action potential")
    ylabel("Voltage")
    tweak()
    xlim(pacelim)

def vis_ct(t=t, y=y):
    plot(t, y["Cai"], **lineplotopt)
    title("Calcium transient")
    ylabel("[Ca]")
    tweak()
    xlim(pacelim)


### Double-pulse protocol

L = vecvclamp(p1p2.protocol)
proto, traj = L[3]
t, y, dy, a = catrec(*traj)


def exp_p1p2(L=L):
    h = []
    for proto, traj in L:
        t, y, dy, a = catrec(*traj)
        h.extend(plot(t, y["V"], **lineplotopt))
    setp(h, color="gray")
    hi = h[len(h) // 2]
    setp(hi, color="black", zorder=10)
    # from IPython.core.debugger import Tracer; Tracer()()
    title("Double-pulse clamping")
    ylabel("Voltage")
    tweak()
    xmin = proto[0][0] - 300
    xmax = t[-1]
    xlim(xmin, xmax)

def vis_p1p2(t=t, y=y):
    markovplot(t, y, comp="fast_sodium", **markovopt)
    tweak_legend()
    title("P1-P2 protocol, I_Na")

### Variable-gap protocol

L = vecvclamp(vargap.protocol)  # list of (protocol, trajectories)
proto, traj = L[3]
t, y, dy, a = catrec(*traj)


def exp_vargap(L=L):
    h = []
    for proto, traj in L:
        t, y, dy, a = catrec(*traj)
        h.extend(plot(t, y["V"], **lineplotopt))
    setp(h, color="gray")
    hi = h[len(h) // 2]
    setp(hi, color="black", zorder=10)
    title("Variable-gap clamping")
    ylabel("Voltage")
    tweak()
    xmin = proto[0][0] - 300
    xmax = getp(hi, "xdata")[-1]
    # from IPython.core.debugger import Tracer; Tracer()()
    xlim(xmin, xmax)

def vis_vargap(t=t, y=y):
    markovplot(t, y, comp="L_type", **markovopt)
    tweak_legend()
    title("Variable gap protocol, I_CaL")


### List of models

def model(name):
    def result():
        plt.title(name)
        axis("off")
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
        # spij(m, n, i, j)
        spij(n, m, j, i)
        panelij()

plt.show()
