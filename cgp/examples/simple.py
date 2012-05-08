"""
A very simple example of a causally cohesive genotype-phenotype study.

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

This very simple example defines everything in one file for illustration. 
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

Large-scale computations will benefit from caching and parallelization.
Examples are given in  
:mod:`~cgp.examples.simple.simple_joblib`, 
:mod:`~cgp.examples.simple.simple_ipython_parallel`, 
:mod:`~cgp.examples.simple.simple_hdfcache`.
"""
# pylint: disable=R0913, R0914, W0142, C0111, W0621

import numpy as np

# Genotype-to-parameter map
def monogenicpar(genotype, hetpar, relvar=0.5, absvar=None):
    """
    Genotype-to-parameter map that assumes one biallelic locus per parameter.
    
    :param genotype: sequence where each item is 0, 1, or 2, 
        denoting the "low" homozygote, the "baseline" heterozygote, and 
        the "high" homozygote, respectively.
    :param recarray hetpar: parameter values for "fully heterozygous" 
        individual.
    :param float relvar: proportion change.
    :param array_like absvar: absolute change, overrides relvar if present
    :return: Record array of parameter values for the given genotype.
    
    Gene/parameter names are taken from the fieldnames of *hetpar*.
    
    Example: Define the baseline parameter values for the full heterozygote, 
    as well as an example genotype whose three loci are high homozygous, 
    heterozygous, and low homozygous, respectively. Mapping this genotype 
    onto parameter space, we see that the parameter values are 150%, 100% and
    50% of their respective baselines, as assumed.
    
    >>> hetpar = np.rec.fromrecords([(0.75, 0.5, 0.25)], 
    ...     names=["a", "b", "theta"])
    >>> genotype = [2, 1, 0]
    >>> monogenicpar(genotype, hetpar)
    rec.array([(1.125, 0.5, 0.125)], 
          dtype=[('a', '<f8'), ('b', '<f8'), ('theta', '<f8')])
    
    Specifying relative or absolute parameter variation.
    
    >>> monogenicpar(genotype, hetpar, relvar = 1.0)
    rec.array([(1.5, 0.5, 0.0)],...
    >>> monogenicpar(genotype, hetpar, absvar = [1.5, 0.5, 0.0])
    rec.array([(2.25, 0.5, 0.25)], ...
    """
    genotype = np.array(genotype)
    if genotype.dtype.names:  # recarray
        genotype = np.array(genotype.item())  # drop names
    result = hetpar.copy().view(float)
    if absvar is None:
        absvar = result * relvar
    else:
        absvar = np.array(absvar).view(float)
    result += (genotype - 1) * absvar
    return result.view(hetpar.dtype, np.recarray)

# Physiological model mapping parameters to raw phenotype
def fitzhugh(y, _t=None, a=0.7, b=0.8, theta=0.08, I=0.0):
    """
    FitzHugh (1969) version of FitzHugh-Nagumo model of nerve membrane
    
    Default parameter values are from FitzHugh (1969), figure 3-2.
    
    :param array_like y: State vector [V, W],where V is the transmembrane 
        potential and W is a "recovery variable".
    :param scalar _t: Time. Ignored, but required by scipy.integrate.odeint.
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
    Vdot = V - V * V * V / 3.0 - W + I # Eq. (3-6) of FitzHugh (1969)
    Wdot = theta * (V + a - b * W)     # Eq. (3-6) of FitzHugh (1969)
    return Vdot, Wdot

# Deriving phenotypes from the physiological model
import scipy.optimize
import numpy.linalg
def eq(func, Y0, disp=False, *args, **kwargs):
    """
    Find equilibrium by minimizing the norm of the rate vector.
    
    :param function func: Function to be minimized.
    :param array_like Y0: Starting point for estimation.
    :param bool disp: Print diagnostics from the minimization?
    :param ``*args, **kwargs``: Additional arguments passed to ``func()``.
    
    >>> eq(fitzhugh, [0, 0])
    array([-1.19942411, -0.62426308])
    """
    def f(Y, *args, **kwargs):
        return numpy.linalg.norm(func(Y, *args, **kwargs))
    return scipy.optimize.fmin(f, Y0, disp=disp, *args, **kwargs)

# Virtual experiment: stimulus-induced action potential
import scipy.integrate
import functools # partial() to simplify function signature
def ap(func, Y0, t0=0.0, stim_period=100.0, stim_amplitude=0.7, 
    stim_duration=1.0, dt=1.0, curr_name="I", *args, **kwargs):
    """
    Stimulus-induced action potential.
    
    :param function func: Right-hand side of ordinary differential equation 
        which can be passed to :func:`scipy.integrate.odeint`.
    :param ndarray Y0: Initial state vector.
    :param float t0: Initial time; helpful in keeping cumulative time for 
        successive action potentials.
    :param float stim_period: Time between start of successive stimuli.
    :param float stim_amplitude: Amplitude of stimulus current.
    :param stim_duration: Duration of stimulus current.
    :param float dt: Time resolution of integration.
    :param str curr_name: Name of the stimulus current argument.
    :param ``*args, **kwargs``: Passed to ``func()``.
    :return: (t, y), where y[i] is state at time t[i].
    
    ..  plot::
        :include-source:
        
        from cgp.examples.simple import *
        t, Y = ap(fitzhugh, [-1.2, -0.6])
        plt.subplot(121)
        plt.plot(t, Y)
        plt.legend(["V", "W"])
        plt.subplot(122)
        plt.plot(*Y.T)
        plt.xlabel("V")
        plt.ylabel("W")
    """
    # Create right-hand sides for two sets of ordinary differential equations,
    # one with stimulus and one without.
    f1 = functools.partial(func, *args, **kwargs)
    kwargs[curr_name] = stim_amplitude
    f0 = functools.partial(func, *args, **kwargs)
    # Make time arrays with step dt but length at least 2
    t0_ = t0 + np.r_[np.arange(0, stim_duration, dt), stim_duration]
    t1_ = t0 + np.r_[np.arange(stim_duration, stim_period, dt), stim_period]
    # Integrate piecewise
    Yout0 = scipy.integrate.odeint(f0, Y0, t0_)
    Yout1 = scipy.integrate.odeint(f1, Yout0[-1], t1_)
    tout = np.concatenate([t0_[:-1], t1_])
    Yout = np.concatenate([Yout0[:-1], Yout1])
    return tout, Yout

def aps(func, Y0, n=5, apfunc=ap, *args, **kwargs):
    """
    Consecutive stimulus-induced action potentials.
    
    Returns a list with one (time, states) tuple per action potential.
    
    You can iterate over the list of tuples like so:
    
    >>> for t, Y in aps(fitzhugh, [0, 0]): # iterating over list of tuples
    ...     pass # e.g. pylab.plot(t, Y)
    
    The list of tuples can be unpacked directly if n is known.
    
    >>> (t0, Y0), (t1, Y1) = aps(fitzhugh, [0, 0], n=2)
    
    Time is reckoned consecutively, not restarting for each action potential.
    
    >>> t0[-1] == t1[0]
    True

    The default parameter values for :func:`fitzhugh` give a regular action 
    potential. Shortening the period can give alternans, whereas an 
    insufficient stimulus will not elicit an action potential.
    
    .. plot::
       
       from cgp.examples.simple import *
       scenarios = dict(Regular=aps(fitzhugh, [0, 0]),
                        Alternans=aps(fitzhugh, [0, 0], stim_period=7.0, dt=0.01),
                        Insufficient=aps(fitzhugh, [0, 0], stim_duration=0.1))
       for i, (k, v) in enumerate(scenarios.items()):
           for t, y in v:
               plt.subplot(len(scenarios), 1, 1 + i)
               plt.plot(t, y)
               plt.title(k)
    """
    result = []
    t0 = 0.0
    for _i in range(n):
        t, Y = apfunc(func, Y0, t0, *args, **kwargs)
        result.append((t, Y))
        t0, Y0 = t[-1], Y[-1]
    return result

# Summarizing phenotypes
from cgp.utils import extrema
def recovery_time(v, t=None, p=0.90):
    """
    Time to ``p*100%`` recovery from first local extremum to initial value.
    
    Intended for action potential duration and the like.
    
    :param array_like t: Time, defaults to [0, 1, 2, ...].
    
    Examples:
    
    >>> recovery_time([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 0])
    11.8...
    >>> recovery_time([10, 0, 20], [100, 101, 102])
    101.45
    >>> recovery_time([10, 0, 20], p=0.5)
    1.25
    """
    if t is None:
        t = np.arange(len(v))
    assert 0 <= p <= 1
    ex = extrema.extrema(v)[:3] # first three extrema including initial value
    # We'll interpolate in a monotone slice of (v, t)
    start, end = ex.index[1], 1 + ex.index[2]
    vi, ti = v[start:end], t[start:end]
    vp = p * ex.value[0] + (1 - p) * ex.value[1] # recovery criterion
    if ex.curv[1] < 0: # recovering from a maximum
        vi = vi[::-1] # interp requires vi to be increasing,
        ti = ti[::-1] # so reverse vi and ti
    return np.interp(vp, vi, ti) # interpolate t as function of v [sic]

# Define baseline parameters
baseline = [("a", 0.7), ("b", 0.8), ("theta", 0.08), ("I", 0.0)]
names, rec = zip(*baseline)
hetpar = np.rec.fromrecords([rec], names=names)

# Enumerate all possible genotypes
nloci = len(baseline)
gts = np.broadcast_arrays(*np.ix_(*([[0, 1, 2]] * nloci)))
gts = np.column_stack([gt.flatten() for gt in gts])

def ph(gt):
    """Ad hoc function to compute the phenotype of a single genotype"""
    par = monogenicpar(gt, hetpar)
    # Must cast par[k] to float for fitzhugh() to work properly
    d = dict((k, float(par[k])) for k in par.dtype.names)
    # Make a new ODE function with overridden defaults
    f = functools.partial(fitzhugh, **d)
    # Keep only the last action potential for statistics
    t, Y = aps(f, Y0=eq(f, [0, 0]))[-1]
    # Aggregating to simpler phenotypes
    apd90 = recovery_time(Y[:, 1], t - t[0])
    return np.rec.fromrecords([(apd90,)], names=["apd90"])

def visualize(result):
    """Trivial example of visualization."""
    import matplotlib.pyplot as plt
    plt.plot(np.sort(result.apd90))
    plt.xlabel("Genotype #")
    plt.ylabel("APD90")
    plt.show()

if __name__ == "__main__":
    # Putting it all together: APD90 for all genotypes
    result = np.concatenate([ph(gt) for gt in gts]).view(np.recarray)
    visualize(result)
