"""Basic example of cGP model."""

import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt

from cgp.gt.genotype import Genotype
from cgp.gt.gt2par import monogenicpar
from cgp.utils.extrema import extrema
from cgp.utils.splom import spij

# Assuming each parameter is governed by one biallelic, additive locus
par0 = np.rec.array([(0.7, 0.8, 0.08, 0.0)], names="a b theta I".split())

# Enumerating all possible genotypes
gt = Genotype(names=par0.dtype.names)

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
    V, W = y
    a, b, theta, I = par
    Vdot = V - V * V * V / 3.0 - W + I # Eq. (3-6) of FitzHugh (1969)
    Wdot = theta * (V + a - b * W)     # Eq. (3-6) of FitzHugh (1969)
    return Vdot, Wdot

def par2ph(par):
    """Parameter-to-phenotype mapping."""
    y0 = np.rec.array([(-0.5, -0.5)], 
        dtype=[("V", float), 
               ("W", float)])
    t = np.arange(0, 1000, 0.1)
    # Use .item() to convert between named and plain arrays
    y = scipy.integrate.odeint(fitzhugh, y0.item(), t, args=(par.item(),))
    y = y.view(y0.dtype, np.recarray)
    return t, y

def ph2agg(t, y, tol=1e-3):
    """Phenotype aggregation."""    
    e = extrema(y.V, withend=False)
    peak = e[e.curv < 0]
    trough = e[e.curv > 0]
    period = np.diff(t[peak.index[-2:]])
    amplitude = peak.value[-1] - trough.value[-1]
    if period and (amplitude > tol):
        return period, amplitude
    else:
        return 0.0, 0.0

if __name__ == "__main__":
    for g in np.array(gt).view(np.recarray):
        par = monogenicpar(g, par0)
        t, y = par2ph(par)
        period, amplitude = ph2agg(t, y)
        spij(3, 3, g.a, g.b)
        plt.plot(period, amplitude, 'ko', ms=1)
        plt.text(period, amplitude, g.theta, 
            ha="center", va="center", size="large")
        plt.title("a = {}, b = {}".format(g.a, g.b))
        if g.a == 2:
            plt.xlabel("Period")
        if g.b == 0:
            plt.ylabel("Amplitude")    
    plt.show()
