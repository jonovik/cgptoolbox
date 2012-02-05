"""
Basic example of cGP study of the FitzHugh-Nagumo model of nerve membrane.

This uses the version in FitzHugh (1969). Default parameter values are as in
figure 3-2 of that paper.

.. plot::
    :width: 400
    :include-source:
    
    from cgp.examples.simple import *
    import scipy.integrate
    
    t = np.arange(0, 50, 0.1)
    y = scipy.integrate.odeint(fitzhugh, [-0.6, -0.6], t)
    plt.plot(t, y)
    plt.legend(["V (transmembrane potential)", "W (recovery variable)"])

References:

* FitzHugh R (1961) 
  :doi:`Impulses and physiological states
  in theoretical models of nerve membrane 
  <10.1016/S0006-3495(61)86902-6>`. 
  Biophysical J. 1:445-466
* FitzHugh R (1969) 
  Mathematical models of excitation and propagation in nerve. 
  In: :isbn:`Biological engineering <978-0070557345>`, 
  ed. H. P. Schwan, 1-85. New York: McGraw-Hill.

In the original version (FitzHugh 1961), the definition of the transmembrane
potential is such that it decreases during depolarization, so that the
action potential starts with a downstroke, contrary to the convention used
in FitzHugh 1969 and in most other work. The equations are also somewhat
rearranged. However, figure 1 of FitzHugh 1961 gives a very good overview of
the phase plane of the model.
"""

import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt

from cgp.gt.genotype import Genotype
from cgp.gt.gt2par import monogenicpar
from cgp.utils.extrema import extrema
from cgp.utils.splom import spij
from cgp.utils.recfun import cbind, restruct
from numpy.distutils.system_info import agg2_info
from cgp.utils.ordereddict import OrderedDict

# Default parameter values from FitzHugh (1969), figure 3-2.
par0 = np.rec.array([(0.7, 0.8, 0.08, 0.0)], names="a b theta I".split())
absvar = np.array([(0.5, 0.5, 0.05, 0.05)])

# Will assume each parameter is governed by one biallelic, additive locus.
genotypes = Genotype(names=par0.dtype.names)

# Physiological model mapping parameters to raw phenotype
def fitzhugh(y, t=None, par=par0):
    """
    FitzHugh (1969) version of FitzHugh-Nagumo model of nerve membrane.
    
    Default parameter values are from FitzHugh (1969), figure 3-2.
    
    :param array_like y: State vector [V, W],where V is the transmembrane 
        potential and W is a "recovery variable".
    :param scalar t: Time. Ignored, but required by scipy.integrate.odeint.
    :param float a, b, theta: Positive constants.
    :param float I: Transmembrane stimulus current.
    
    References: see :mod:`module docstring <cgp.examples.basic>`.
    """
    V, W = y
    a, b, theta, I = par
    Vdot = V - V * V * V / 3.0 - W + I # Eq. (3-6) of FitzHugh (1969)
    Wdot = theta * (V + a - b * W)     # Eq. (3-6) of FitzHugh (1969)
    return Vdot, Wdot

def gt2par(gt):
    """Genotype-to-parameter mapping."""
    return monogenicpar(gt, hetpar=par0, absvar=absvar)

def par2ph(par):
    """Parameter-to-phenotype mapping."""
    y0 = np.rec.array([(-0.5, -0.5)], dtype=[("V", float), ("W", float)])
    t = np.rec.fromarrays([np.arange(0, 1000, 0.1)], names="t")
    # Use .item() to convert between named and plain arrays
    y = scipy.integrate.odeint(fitzhugh, y0.item(), t.t, args=(par.item(),))
    # Combine t and fields of y into one recarray with array-valued fields
    return restruct(cbind(t, y.view(y0.dtype)))

def ph2agg(ph, tol=1e-3):
    """Phenotype aggregation."""
    e = extrema(ph["V"], withend=False)
    peak = e[e.curv < 0]
    trough = e[e.curv > 0]
    if len(peak.index) >= 2:
        t0, t1 = ph["t"].squeeze()[peak.index[-2:]]
        period = t1 - t0
        amplitude = peak.value[-1] - trough.value[-1]
        if amplitude < tol:
            period = amplitude = 0.0
    else:
        period = amplitude = 0.0
    return np.rec.fromrecords([(period, amplitude)], 
        names=["period", "amplitude"])


# Various options for connecting the pipeline pieces

def stepwise():
    """Processing all genotypes for one step at a time."""
    
    def cat(arrays):
        return np.concatenate(arrays, axis=-1)
    
    gt = np.array(genotypes)
    par = cat([monogenicpar(g, hetpar=par0, absvar=absvar) for g in gt])
    ph = cat([par2ph(p) for p in par])
    agg = cat([ph2agg(p) for p in ph])
    
    # Scatterplot matrix of aggregated phenotypes vs genotypes
    for i, ai in enumerate(agg.dtype.names):
        for j, gj in enumerate(gt.dtype.names):
            spij(len(agg.dtype.names), len(gt.dtype.names), i, j)
            plt.plot(gt[gj], agg[ai], 'o')
            plt.axis([-0.5, 2.5, 0, agg[ai].max() * 1.1])
            if i == len(agg.dtype) - 1:
                plt.xlabel(gj)
            if j == 0:
                plt.ylabel(ai)
    plt.show()

def genotypewise():
    """Passing one genotype at a time through all steps of the workflow."""
    for gt in genotypes:
        gt = gt.view(np.recarray)
        par = monogenicpar(gt, par0)
        ph = par2ph(par)
        agg = ph2agg(ph).view(np.recarray)
        
        print agg
        
        # Scatterplot matrix of aggregated phenotypes vs genotypes
        for i, ai in enumerate(agg.dtype.names):
            for j, gj in enumerate(gt.dtype.names):
                spij(len(agg.dtype.names), len(gt.dtype.names), i, j)
                plt.plot(gt[gj], agg[ai], 'ko')
                if i == len(agg.dtype) - 1:
                    plt.xlabel(gj)
                if j == 0:
                    plt.ylabel(ai)
    plt.show()

def functional(gt, gt2par, par2ph, ph2agg):
    gt = list(gt)
    par = [gt2par(i) for i in gt]
    ph = [par2ph(i) for i in par]
    agg = [ph2agg(i) for i in ph]
    return dict(gt=gt, par=par, ph=ph, agg=agg)    

def hdfcaching():
    """Auto-cache/save results to HDF."""
    from cgp.utils.hdfcache import Hdfcache
    filename = "/home/jonvi/hdfcache.h5"
    hdfcache = Hdfcache(filename)
    
    pipeline = [hdfcache.cache(i) for i in gt2par, par2ph, ph2agg]
    
    with hdfcache:
        for i in genotypes:
            for func in pipeline:
                i = func(i)
    
    import tables as pt
    
    with pt.openFile(filename) as f:
        gt = f.root.gt2par.input[:]
        agg = f.root.ph2agg.output[:]
        for i, ai in enumerate(agg.dtype.names):
            for j, gj in enumerate(gt.dtype.names):
                spij(len(agg.dtype.names), len(gt.dtype.names), i, j)
                plt.plot(gt[gj], agg[ai], 'ko')
                if i == len(agg.dtype) - 1:
                    plt.xlabel(gj)
                if j == 0:
                    plt.ylabel(ai)
                xmin, xmax, ymin, ymax = plt.axis()
                plt.axis([xmin-0.5, xmax+0.5, -0.1*(ymax-ymin), 1.1*ymax])
    plt.show()
    
    import os
    os.system("h5ls -r " + filename)

def clusterjob():
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="""Basic example of the building blocks of a cGP study.\n\n
        The top-level orchestration can be done in many different ways, 
        depending on personal preference and on the need for features such as 
        storage, caching, memory management or parallelization. A few examples 
        are offered here. To run them, pass one or more arguments such as 
        --stepwise.""")
    options = stepwise, genotypewise, functional, hdfcaching, clusterjob
    for func in options:
        parser.add_argument("--" + func.__name__, help=func.__doc__, 
            action="store_const", const=func)
    args = vars(parser.parse_args())
    for func in args.values():
        if func:
            print "Running '{}'...".format(func.__name__)
            func()
