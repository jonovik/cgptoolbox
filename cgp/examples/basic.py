"""
Basic example of cGP study of the FitzHugh-Nagumo model of nerve membrane.

.. plot::
    :width: 400
    :include-source:
        
    from cgp.examples.basic import fitzhugh
    import scipy.integrate
    
    t = np.arange(0, 50, 0.1)
    y = scipy.integrate.odeint(fitzhugh, [-0.6, -0.6], t)
    plt.plot(t, y)
    plt.legend(["V (transmembrane potential)", "W (recovery variable)"])

..  plot::
    
    from cgp.examples.basic import cgpstudy
    cgpstudy()

The code in this file is ready to run, but is primarily written to be *read*.
It exemplifies the building blocks of a cGP model study:

..  list-table::
    :header-rows: 1

    * * Component
      * Description
      * Example
    * * Generation of genotypes
      * Genotypic variation whose phenotypic effects are to be estimated.
      * Full enumeration feasible if the genotype space is small.
        Other options include pedigrees or statistical experimental designs.
    * * Genotype-to-parameter map
      * Lumping together lower-level physiology such as gene regulation
      * Maximum conductances of ion channels
    * * Parameter-to-phenotype map
      * Physiological model
      * Heart cell electrophysiology
    * * Virtual experiment
      * Manipulating the physiological model
      * Pacing with electrical stimuli
    * * Aggregate phenotypes
      * Summarizing the raw model output
      * Action potential duration
    * * Analysis, summary, visualization
      *
      *

This very simple example defines most components in one file for illustration. 
Real applications will typically draw upon existing code libraries, model 
repositories and databases. Linking these together may require a lot of code, 
a complication which we ignore here to make the main concepts stand out more
clearly.

Our example system is the :func:`FitzHugh-Nagumo model <fitzhugh>` of an 
excitable heart muscle cell. It has four parameters, one of which represents 
a stimulus current. Virtual pacing of the cell can be implemented by setting 
the stimulus current to a nonzero value at regular intervals. The resulting 
"action potential" (time-course of transmembrane voltage) is often 
characterized by its duration, for instance APD90 for the action potential 
duration to 90 % repolarization.

Here, we assume that each parameter is governed by a single gene of which 
there are two alleles. This trivial genotype-to-parameter map is a caricature 
in the absence of actual data, but is a minimal assumption to make the model 
amenable to cGP analysis.

Here we can compute phenotypes for all 27 genotypes (three loci with three 
possibilities each gives 
aa bb cc, aa bb Cc, aa bb CC, aa Bb cc, ..., AA BB CC). In higher-dimensional 
cases, the genotype space must be sampled according to some experimental 
design, possibly constrained by pedigree information.


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
# pylint: disable=W0621

import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt

from cgp.gt.genotype import Genotype
from cgp.gt.gt2par import monogenicpar
from cgp.utils.extrema import extrema
from cgp.utils.splom import spij
from cgp.utils.recfun import cbind, restruct

# Default parameter values from FitzHugh (1969), figure 3-2.
par0 = np.rec.array([(0.7, 0.8, 0.08, 0.0)], names="a b theta I".split())
absvar = np.array([(0.5, 0.5, 0.05, 0.05)])

# Will assume each parameter is governed by one biallelic, additive locus.
genotypes = Genotype(names=par0.dtype.names)

# Physiological model mapping parameters to raw phenotype
def fitzhugh(y, t=None, par=par0):  # pylint: disable=W0613
    """
    FitzHugh (1969) version of FitzHugh-Nagumo model of nerve membrane.
    
    Default parameter values are from FitzHugh (1969), figure 3-2.
    
    :param array_like y: State vector [V, W],where V is the transmembrane 
        potential and W is a "recovery variable".
    :param scalar t: Time. Ignored, but required by scipy.integrate.odeint.
    :param float a, b, theta: Positive constants.
    :param float I: Transmembrane stimulus current.
    
    References: see :mod:`module docstring <cgp.examples.basic>`.
    
    .. seealso:: :class:`cgp.virtexp.examples.Fitz`.
    """
    V, W = y
    a, b, theta, I = par.item()
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
    y = scipy.integrate.odeint(fitzhugh, y0.item(), t.t, args=(par,))
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

def summarize(gt, agg):
    """Scatterplot matrix of aggregated phenotypes vs genotypes."""
    for i, ai in enumerate(agg.dtype.names):
        for j, gj in enumerate(gt.dtype.names):
            spij(len(agg.dtype.names), len(gt.dtype.names), i, j)
            plt.plot(gt[gj], agg[ai], 'o')
            xmin, xmax, ymin, ymax = plt.axis()
            ypad = 0.1 * (ymax - ymin)
            plt.axis([xmin - 0.5, xmax + 0.5, -ypad, ymax + ypad])
            plt.axis()
            if i == len(agg.dtype) - 1:
                plt.xlabel(gj)
            if j == 0:
                plt.ylabel(ai)

def pad_plot():
    """Adjust axis limits to fully show markers."""
    for ax in plt.gcf().axes:
        plt.axes(ax)
        ymin, ymax = plt.ylim()
        ypad = 0.1 * (ymax - ymin)
        plt.axis([-0.5, 2.5, -ypad, ymax + ypad])

def cgpstudy():
    """
    Basic example of connecting the building blocks of a cGP study.
    
    This top-level orchestration can be done in many different ways, depending 
    on personal preference and on the need for features such as storage, 
    caching, memory management or parallelization. Some examples are given in 
    :mod:`cgp.examples.hpc`.
    """
    from numpy import concatenate as cat

    gt = np.array(genotypes)
    par = cat([monogenicpar(g, hetpar=par0, absvar=absvar) for g in gt])
    ph = cat([par2ph(p) for p in par])
    agg = cat([ph2agg(p) for p in ph])
    
    summarize(gt, agg)
    plt.show()


if __name__ == "__main__":
    cgpstudy()
