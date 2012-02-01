"""Basic example of cGP model."""

import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt

from cgp.utils.extrema import extrema
from cgp.gt.genotype import Genotype
from cgp.gt.gt2par import monogenicpar

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
    t = np.arange(0, 100, 0.1)
    # Use .item() to convert between named and plain arrays
    y = scipy.integrate.odeint(fitzhugh, y0.item(), t, args=(par.item(),))
    y = y.view(y0.dtype, np.recarray)
    return t, y

def ph2agg(t, y, tol=1e-3):
    """Phenotype aggregation."""    
    e = extrema(y.V, withend=False)
    peak = e[e.curv < 0]
    trough = e[e.curv > 0]
    if abs(peak.value[-1] - trough.value[-1]) > tol:
        period = np.diff(t[peak.index[-2:]])
        amplitude = peak.value[-1] - trough.value[-1]
        return period, amplitude
    else:
        return np.inf, 0.0

i = gt[0]
par = monogenicpar(i, par0)
ph = par2ph(par)
agg = ph2agg(ph)

par = [monogenicpar(i, par0) for i in gt]
ph = [par2ph(i) for i in par]
agg = [ph2agg(i) for i in ph]



par0.I = 1
t, y = ph = par2ph(par0)
agg = ph2agg(t, y)
print agg
for k in y.dtype.names:
    plt.plot(t, y[k], label=k)
plt.legend()
plt.show()

#t = np.arange(0, 50, 0.1)
#y = scipy.integrate.odeint(fitzhugh, [-0.6, -0.6], t)
#plt.plot(t, y)
#plt.legend(["V (transmembrane potential)", "W (recovery variable)"])
#plt.show()
