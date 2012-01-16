"""
Causally cohesive genotype-phenotype study: FitzHugh-Nagumo nerve membrane model

The code in this file is ready to run, but is primarily written to be *read*.
It exemplifies the building blocks of a cGP model study:

* Genotype-to-parameter map
* Parameter-to-phenotype map (physiological model)
* Virtual experiment (manipulating the physiological model)
* Aggregate phenotypes (summarizing the raw model output)
* Analysis, summary, visualization

You may explore the code in several ways:

* Generate HTML help with `pydoc -p <http://docs.python.org/library/pydoc>`_

* Stepping through the example in IPython, using::
  
  %run -d cgpdemo --demo

* Interactive coding in IPython::
    
    from cgpdemo import *
    PH = cgpstudy()

* ``%whos`` will list the functions defined.
* See individual docstrings for details, e.g. :class:`~ap_cvode.Bond.ap`.
"""

import numpy as np

# Genotype-to-parameter map
def monogenicpar(genotype, hetpar, relvar=0.5, absvar=None):
    """
    Genotype-to-parameter map that assumes one biallelic locus per parameter
    
    :param genotype: sequence where each item is 0, 1, or 2, 
        denoting the "low" homozygote, the "baseline" heterozygote, and 
        the "high" homozygote, respectively. Alternatively, each item can 
        be a 2-tuple where each item is 0 or 1.
    :param recarray hetpar: parameter values for "fully heterozygous" 
        individual
    :param float relvar: proportion change
    :param array_like absvar: absolute change, overrides relvar if present
    :return recarray: Parameter values for the given genotype.
    
    Gene/parameter names are taken from the fieldnames of *hetpar*.
    Thus, the result has the same dtype (Numpy data type) as the genotype array.
    
    Creating a record array of baseline parameter values 
    (corresponding to the fully heterozygous genotype).
    
    >>> baseline = [("a", 0.25), ("b", 0.5), ("theta", 0.75), ("I", 1.0)]
    >>> dtype = [(k, float) for k, v in baseline]
    >>> hetpar = np.array([v for k, v in baseline]).view(dtype, np.recarray)
    
    Parameter values for four "loci" with 
    (low homozygote, heterozygote, heterozygote, high homozygote).
    
    >>> genotype = [[0, 0], [0, 1], [1, 0], [1, 1]]
    >>> monogenicpar(genotype, hetpar)
    rec.array([(0.125, 0.5, 0.75, 1.5)], 
          dtype=[('a', '<f8'), ('b', '<f8'), ('theta', '<f8'), ('I', '<f8')])
    
    Specifying relative or absolute parameter variation.
    
    >>> monogenicpar(genotype, hetpar, relvar = 1.0)
    rec.array([(0.0, 0.5, 0.75, 2.0)],...
    >>> monogenicpar(genotype, hetpar, absvar = [0.125, 0.25, 0.5, 0.75])
    rec.array([(0.125, 0.5, 0.75, 1.75)],...
    """
    genotype = np.array(genotype)
    if genotype.ndim == 2:
        allelecount = np.array(genotype).sum(axis=1)
    else:
        allelecount = genotype
    result = hetpar.copy().view(float)
    if absvar is None:
        absvar = result * relvar
    else:
        absvar = np.array(absvar).view(float)
    result += (allelecount - 1) * absvar
    return result.view(hetpar.dtype, np.recarray)

# Physiological model mapping parameters to raw phenotype
def fitzhugh(Y, t=0.0, a=0.7, b=0.8, theta=0.08, I=0.0):
    """
    FitzHugh (1969) version of FitzHugh-Nagumo model of nerve membrane
    
    References:
    FitzHugh R (1969) 
    Mathematical models of excitation and propagation in nerve. 
    In: Biological engineering, ed. H. P. Schwan, 1-85. New York: McGraw-Hill.
    
    Default parameter values are from FitzHugh (1969), figure 3-2.
    
    * V = Y[0] = membrane potential
    * W = Y[1] = recovery variable
    * a, b, theta are positive constants
    * I is the membrane current

    Integration over time.
    
    .. plot::
        :context:
        :include-source:
        :nofigs:
        
        >>> from cgpdemo import *
        >>> import scipy.integrate
        >>> t = np.arange(0, 50, 0.1)
        >>> Y = scipy.integrate.odeint(fitzhugh, [-0.6, -0.6], t)
    
    .. plot::
        :context:
        
        plt.clf()
        plt.plot(t, Y)
    
    Showing only the first and last two states.
    
    >>> ix = np.r_[0:2, -2:0]
    >>> Y[ix]
    array([[-0.6       , -0.6       ],
           [-0.59280103, -0.59534608],
           [-1.2045089 , -0.62472546],
           [-1.20423976, -0.62476208]])
    
    The original version of the model is        
    FitzHugh R (1961) 
    Impulses and physiological states in theoretical models of nerve membrane. 
    Biophysical J. 1:445-466
    
    In the original version, the definition of the transmembrane potential is 
    such that it decreases during depolarization, so that the action potential 
    starts with a downstroke, contrary to the convention used in FitzHugh 1969 
    and in most other work. The equations are also somewhat rearranged. 
    However, figure 1 of FitzHugh 1961 gives a very good overview of the phase 
    plane of the model.
    """
    V, W = Y
    Vdot = V - V * V * V / 3.0 - W + I # Eq. (3-6) of FitzHugh (1969)
    Wdot = theta * (V + a - b * W)     # Eq. (3-6) of FitzHugh (1969)
    return Vdot, Wdot

# Plot phase plane for FitzHugh-Nagumo model

def phaseplane():
    """
    Phase plane plot for the FitzHugh-Nagumo model
    
    Produces a plot similar to Figure 3.2 in FitzHugh (1969)

    .. plot::
    
       from cgpdemo import *
       phaseplane()
    """
    import pylab

    X, Y, U, V = [], [], [], []
    for w in np.linspace(-1,1.5,50):
        for v in np.linspace(-2.5,2,50):
            X.append(v)
            Y.append(w)
            u, v = fitzhugh([v,w])
            U.append(u)
            V.append(v)
    pylab.quiver(X,Y,np.sign(U),np.sign(V),np.sign(U)+2*np.sign(V)) #plot phase-plane omitting magnitudes
    pylab.hold(True)

    for s in np.linspace(0.6,0.7,10):
        t, Y = ap(fitzhugh, [-1.2, -0.6],stim_curr=s)
        pylab.plot(*Y.T)

    pylab.xlabel('Membrane potential (V)')
    pylab.ylabel('Recovery variable (W)')
    pylab.title('Phase plane and some trajectories for the FitzHugh-Nagumo model')
    pylab.show()
    return

# Deriving phenotypes from the physiological model
import scipy.optimize
import numpy.linalg
def eq(func, Y0, disp=False, *args, **kwargs):
    """
    Find equilibrium by minimizing the norm of the rate vector.
    
    :param bool disp: Prints diagnostics from the minimization?
    
    >>> eq(fitzhugh, [0, 0])
    array([-1.19942411, -0.62426308])
    """
    def f(Y, *args, **kwargs):
        return numpy.linalg.norm(func(Y, *args, **kwargs))
    return scipy.optimize.fmin(f, Y0, disp=disp, *args, **kwargs)

# Virtual experiment: stimulus-induced action potential
import scipy.integrate
import functools # partial() to simplify function signature
def ap(func, Y0, t0=0.0, stim_per=100.0, stim_curr=0.7, stim_dur=1.0, 
    dt=1.0, curr_name="I", *args, **kwargs):
    """
    Stimulus-induced action potential
    
    Return an array of time-points and an array of states at each time-point.
    Experiments that run consecutive action potentials can keep cumulative time
    by passing the last time-point returned by the current action potential as
    the initial time t0 for the next.
    
    >>> t, Y = ap(fitzhugh, [-1.2, -0.6])
    >>> Y.max(axis=0)
    array([ 1.69664227,  0.98256442])
    >>> Y.min(axis=0)
    array([-2.00558297, -0.62509846])
    
    Plotting trajectory vs. time or in state space.
    
    .. plot::
       :include-source:
       
       from cgpdemo import *
       t, Y = ap(fitzhugh, [-1.2, -0.6])
       plt.plot(t, Y)
       plt.show()
       plt.plot(*Y.T)
    """
    # Make two functions with different defaults
    f1 = functools.partial(func, *args, **kwargs)
    kwargs[curr_name] = stim_curr
    f0 = functools.partial(func, *args, **kwargs)
    # Make time arrays with step dt but length at least 2
    n0 = max(2, round(stim_dur / dt))
    n1 = max(2, round((stim_per - stim_dur) / dt))
    t0_ = t0 + np.linspace(0, stim_dur, n0)
    t1_ = t0 + np.linspace(stim_dur, stim_per, n1)
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
       :include-source:
       :nofigs:
       :context:
       
       >>> from cgpdemo import *
       >>> regular = aps(fitzhugh, [0, 0])
       >>> alternans = aps(fitzhugh, [0, 0], stim_per=7.0)
       >>> insufficient = aps(fitzhugh, [0, 0], stim_dur=0.1)

    Plotting:
    
    .. plot::
       :include-source:
       :context:
       
       for L in regular, alternans, insufficient:
           for t, Y in L:        
               plt.plot(t, Y)
    
    (Doctests to guard against accidental changes.)
    
    >>> ix = np.r_[0:2, -2:0]
    >>> t1[ix]
    array([ 100.        ,  101.        ,  198.98979592,  200.        ])
    >>> Y1[ix]
    array([[-1.19940804, -0.62426004],
           [-0.49837833, -0.59772333],
           [-1.19940805, -0.62426004],
           [-1.19940805, -0.62426004]])
    """
    result = []
    t0 = 0.0
    for i in range(n):
        t, Y = apfunc(func, Y0, t0, *args, **kwargs)
        result.append((t, Y))
        t0, Y0 = t[-1], Y[-1]
    return result

# Summarizing phenotypes
from utils import extrema
def recovery_time(v, t=None, p=0.90):
    """
    Time to ``p*100%`` recovery from first local extremum to initial value.
    
    Intended for action potential duration and the like.
    
    :param array_like t: Time, defaults to [0, 1, 2, ...].
    
    Examples:
    
    >>> recovery_time([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 0])
    11.800...
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

# Putting it all together: APD90 for all genotypes
from utils.hdfcache import Hdfcache
hdfcache = Hdfcache("cgpdemo.h5")
def cgpstudy():
    """A simple causally cohesive genotype-phenotype model study"""
    # Define baseline parameters
    baseline = [("a", 0.7), ("b", 0.8), ("theta", 0.08), ("I", 0.0)]
    dtype = [(k, float) for k, v in baseline]
    hetpar = np.array([v for k, v in baseline]).view(dtype, np.recarray)
    # Enumerate all possible genotypes
    nloci = len(baseline)
    gts = np.broadcast_arrays(*np.ix_(*([[0, 1, 2]] * nloci)))
    gts = np.column_stack([gt.flatten() for gt in gts])
    
    # Iterate over genotypes:
    @hdfcache.cache
    def ph(gt, dtype=[("apd90", float)]):
        """Ad hoc function to compute the phenotype of a single genotype"""
        par = monogenicpar(gt, hetpar)
        # Must cast par[k] to float for fitzhugh() to work properly
        d = dict((k, float(par[k])) for k in par.dtype.names)
        # Make a new ODE function with overridden defaults
        f = functools.partial(fitzhugh, **d)
        # Keep only the last action potential for statistics
        t, Y = aps(f, Y0=eq(f, [0, 0]))[-1]
        # Aggregating to simpler phenotypes
        apd90 = recovery_time(Y[:,1], t - t[0])
        return np.array((apd90,), dtype=dtype)
    
    PH = np.concatenate([ph(gt) for gt in gts]).view(np.recarray)
    # Visualization (example intended for interactive use, e.g. under IPython)
    from pylab import plot, xlabel, ylabel, show
    plot(np.sort(PH.apd90))
    xlabel("Genotype #")
    ylabel("APD90")
    show()
    return PH

if __name__ == "__main__":
    from optparse import OptionParser
    usage = """usage: %prog [options]
        With no options or -v, run doctests. 
        With --demo, run a simple cGP model study."""
    parser = OptionParser(usage=usage)
    # Run time-consuming demo only if requested
    parser.add_option("-d", "--demo", action="store_true", 
        help="Run an example cGP analysis")
    # The doctest module expects a --verbose option
    parser.add_option("-v", "--verbose", action="store_true", 
        help="Run doctests with verbose output")
    (options, args) = parser.parse_args()
    if options.demo:
        with hdfcache:
            PH = cgpstudy()
    else:
        import doctest
        doctest.testmod(optionflags=doctest.ELLIPSIS)
