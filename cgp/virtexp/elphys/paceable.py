"""
Virtual experiments for cellular electrophysiology.

These protocols assume that the :wiki:`transmembrane potential` is a variable 
named *V* in the model. (If the transmembrane potential is named differently, 
use the *rename* argument to the 
:meth:`~cgp.physmod.cellmlmodel.Cellmlmodel` constructor.)

Many models of cellular electrophysiology, such as the 
:cellml:`Bondarenko 
<11df840d0150d34c9716cd4cbdd164c8/bondarenko_szigeti_bett_kim_rasmusson_2004_apical>`
and 
:cellml:`Ten Tusscher 
<e946a72663bdf17ef6752980a0232351/tentusscher_noble_noble_panfilov_2004_a>` 
models, have hardcoded a protocol of regular
pacing, which must be disabled to apply other protocols such as regular pacing 
(:doi:`Cooper et al. 2011 <10.1016/j.pbiomolbio.2011.06.003>`).

Here, regular pacing is assumed to be governed by the following parameters:

* *stim_amplitude* : Magnitude of the stimulus current.
* *stim_period* : Interval between the beginnings of successive stimuli.

Some models have a parameter called *stim_start*, running the model unpaced 
for some time before the stimulus starts. Here it is assumed that any such 
delay is set to zero.

The implementation of voltage clamp experiments assumes that the model object 
has a 
:meth:`~cgp.cvodeint.namedcvodeint.Namedcvodeint.clamp` method.
"""  # pylint: disable=C0301

from __future__ import division # 7 / 4 = 1.75 rather than 1
from . import ap_stats
from collections import namedtuple, deque
from pysundials import cvode
import ctypes
import numpy as np



class Paceable(object):
    """
    Mixin class for pacing electrophysiological models.
    
    This protocol is applicable to any model with a transmembrane potential 
    *V* and stimulus parameters *stim_amplitude* and *stim_period*.
    
    To use, define a class inheriting both from :class:`Paceable` and 
    :class:`~cgp.physmod.cellmlmodel.Cellmlmodel`.
    
    >>> from cgp.virtexp.elphys import Cellmlmodel, Paceable
    >>> class Model(Cellmlmodel, Paceable):
    ...    pass
    >>> model = Model(workspace="bondarenko_szigeti_bett_kim_rasmusson_2004")
    
    The model identifiers (workspace, exposure, workspace) can be cumbersome 
    to remember, and it may be convenient to define new defaults in the class 
    constructor. See mod:`cgp.virtexp.elphys.examples` for examples.
    """

    def ap(self, p_repol=(0.25, 0.5, 0.75, 0.9), 
        ignore_flags=False, rootfinding=False):
        r"""
        Simulate action potential triggered by stimulus current.
        
        :param array_like p_repol: compute action potential duration to 
           :math:`p_{repol} \times 100\%` repolarization
        :param bool ignore_flags: disable sanity checking of the shape of the 
           action potential
        :param bool rootfinding: use CVODE's rootfinding facilities to 
           compute action potential statistics more accurately
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> cell = Bond()
        >>> t, y, stats = cell.ap(p_repol=(0.25, 0.5))
        
        Stimulation starts at time 0.0, overriding any default in the CellML:
        
        >>> cell.pr.stim_start
        array([ 0.])
        
        Calling :meth:`ap` again resumes from the previous state, resetting 
        time to 0:
        
        >>> t1, Y1, stats1 = cell.ap(p_repol=(0.25, 0.5))
        >>> t1[0]
        0.0
        
        Parameters governing the stimulus setup:
        
        >>> [(s, cell.pr[s]) for s in cell.pr.dtype.names
        ...     if s.startswith("stim_")]
        [('stim_start', array([ 0.])), 
         ('stim_end', array([ 100000.])),
         ('stim_period', array([ 71.43])),
         ('stim_duration', array([ 0.5])),
         ('stim_amplitude', array([-80.]))]
        
        >>> from pprint import pprint # platform-independent order of dict items
        >>> pprint(stats)
        {'amp': 115.44...,
         'base': -82.4201999...,
         'caistats': {'amp': 0.525...},
         'decayrate': array([ 0.223...
         'peak': 33.021572...,
         't_repol': array([  3.31012...,   5.125...]),
         'ttp': 1.844189...}
        
        Module-level variables are shared between instances!
        
        >>> b0 = Bond(); b1 = Bond()
        >>> original_y00 = b0.model.y0[0]
        >>> b0.model.y0 is b1.model.y0
        True
        >>> b0.model.y0[0] = 12345
        >>> b1.model.y0[0]
        12345.0
        >>> b1.y0r.V = 54321
        >>> b0.y0r.V
        array([ 54321.])
        
        Reset the defaults so we don't mess up other doctests:
        
        >>> b0.y0r.V = original_y00
        >>> assert b0.y0r.V == b1.model.y0[0] == original_y00

        To temporarily modify initial state or parameters, use 
        :meth:`~cvodeint.namedcvodeint.Namedcvodeint.autorestore`.
        
        >>> with cell.autorestore(V=-60, stim_period=50):
        ...     t, y, stats = cell.ap(p_repol=(0.25, 0.5))
        >>> t[-1]
        50.0
        >>> pprint(stats)
        {'amp': 87.74...,
         'base': -60.0,
         'caistats': {'amp': 0.182...},
         'decayrate': array([ 0.369...
         'peak': 27.74...,
         't_repol': array([  2.550...,   3.648...]),
         'ttp': 1.327...}
        """
        if rootfinding:
            return self._ap_with_rootfinding(p_repol, ignore_flags)
        else:
            return self._ap_without_rootfinding(p_repol, ignore_flags)
    
    def _ap_with_rootfinding(self, p_repol=(0.25, 0.5, 0.75, 0.9), 
        ignore_flags=False):
        r"""        
        Simulate action potentials using CVODE's rootfinding facilities.
        
        Arguments are as for :meth:`ap`.
        
        Effect of tolerances:
        
        .. plot::
            :width: 300
        
            from cgp.virtexp.elphys.examples import Bond
            for tol in 1e-6, 1e-2:
                t, y, stats = Bond(reltol=tol).ap()
                plt.plot(t, y.V, '.-', label="reltol={:.0e}".format(tol))
                i = stats["i"]
                plt.plot(t[i], y.V[i], 'ro')
            plt.xlim(1.6, 2.2)
            plt.ylim(32, 33.1)
            plt.legend()
        
        If the action potential does not have the expected shape, for example 
        due to an insufficient stimulus, an exception explains that the 
        integrator returned an unexpected flag.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> bond = Bond()
        >>> with bond.autorestore(stim_amplitude=0):
        ...     t, y, stats = bond.ap(rootfinding=True)
        Traceback (most recent call last):
        CvodeException: CVode returned CV_TSTOP_RETURN
        
        Note that we can suppress the exception and still have reasonable output.
        
        >>> with bond.autorestore(stim_amplitude=0):
        ...     t, y, stats = bond.ap(rootfinding=True, ignore_flags=True)
        """
        
        result = [] # list to keep results from integration subintervals
        
        # integrate over stimulus
        result.append(self.integrate(t=[0, self.pr.stim_duration], nrtfn=0, 
            assert_flag=cvode.CV_TSTOP_RETURN, ignore_flags=ignore_flags))
        
        # integrate from stimulus to peak
        j_peak = 0 # index to "result" item ending with peak
        # make sure we don't stop at a minor peak at end of stimulus
        # repeat until we are at an extremum with V > 0
        while True:
            j_peak += 1
            result.append(self.integrate(t=self.pr.stim_period, 
                nrtfn=1, g_rtfn=self.ydoti("V"), assert_flag=cvode.CV_ROOT_RETURN, 
                ignore_flags=ignore_flags))
            _tj, yj, flagj = result[-1]
            if (flagj == cvode.CV_TSTOP_RETURN) or (yj.V[-1] > 0):
                break
        
        # compute repolarization thresholds
        Vmin = result[0][1][0].V # 1st integration, 2nd return var, 1st step
        # was: Vmax = result[-1][1][-1].V
        Vmax = yj.V[-1] # last integration, 2nd return var, last step
        V_repol = Vmax - p_repol * (Vmax - Vmin)        
        
        # integrate to each repolarization threshold in turn
        g_rtfn = self.repol()
        result += [self.integrate(t=self.pr.stim_period,
            nrtfn=1, g_rtfn=g_rtfn, g_data=ctypes.byref(ctypes.c_float(x)),
            assert_flag=(cvode.CV_ROOT_RETURN, cvode.CV_TSTOP_RETURN), 
            ignore_flags=ignore_flags)
            for x in V_repol]
        
        # If time has run out, flag is None but flagj is CV_TSTOP_RETURN
        if result[-1][-1] == cvode.CV_ROOT_RETURN:
            # integrate from repolarization to next stimulus:
            result.append(self.integrate(t=self.pr.stim_period, nrtfn=0, 
                assert_flag=cvode.CV_TSTOP_RETURN, 
                ignore_flags=ignore_flags))        
        
        # drop intervals where flag is None; those were past t_stop
        result = [res for res in result if res[-1] is not None]
        t, Y, _flag = zip(*result) # (t, Y, flag), where each is a tuple
        
        # The items of the tuples refer to these intervals, assuming the
        # default p_repol specifying four thresholds:
        # 0) stimulus
        # ...+ possibly more here, waiting for some extremum V >= 0...
        # 1) stimulus to peak,
        # 2) peak to first repolarization threshold
        # 3) first to second repolarization threshold
        # 4) second to third repolarization threshold
        # 5) third to fourth repolarization threshold
        # 6) fourth repolarization threshold to just before next stimulus
        
        stats = {"base": Y[0][0].V, "ttp": t[j_peak][-1], 
            "peak": Y[j_peak][-1].V, "p_repol": p_repol}
        # summarize "result" items for repolarization (items _after_ j_peak)
        stats["t_repol"] = np.array([ti[-1]
            for ti, _yi, flagi in result[j_peak + 1:]
            if flagi == cvode.CV_ROOT_RETURN])
        # keep indices for j_peak and later items
        stats["i"] = np.cumsum([len(ti) for ti in t])[j_peak:-1] - 1        
        # concatenation converts recarray to ndarray, so need to convert back
        Y = np.concatenate(Y).view(np.recarray)
        t = np.concatenate(t)
        # just use (arg)max if rootfinding didn't work:
        if len(stats["i"]) == 0:
            stats["i"] = [np.argmax(Y.V)] # ensures that t[stats["i"]] is array
            stats["peak"] = Y.V[stats["i"]]
            stats["ttp"] = t[stats["i"]]
            stats["t_repol"] = []
            stats["p_repol"] = []
        try:
            stats["decayrate"] = ap_stats.apd_decayrate(stats, p=[0.25, 0.9])
        except (ValueError, IndexError):
            stats["decayrate"] = np.nan
        stats["amp"] = stats["peak"] - stats["base"]
        
        assert t[stats["i"]][0] == stats["ttp"]
        ti = t[stats["i"]][1:]
        trepol = stats["t_repol"]
        maxlen = min(len(ti), len(trepol))
        assert (ti[:maxlen]==trepol[:maxlen]).all()
        try:
            sc = stats["caistats"] = ap_stats.apd(t, Y.Cai)
            # don't bother to report Ca decay rate for very small oscillations 
            if (sc["amp"] / sc["peak"]) < 1e-3:
                sc["decayrate"] = np.nan
            else:
                sc["decayrate"] = ap_stats.apd_decayrate(sc, p=[0.25, 0.9])
        except AttributeError: # model may not have a state variable called Cai
            pass
        return t, Y, stats
    
    def repol(self):
        """
        Difference between current V and repolarization threshold.
        
        .. seealso:: :func:`pysundials.cvode.CVodeRootInit`
        
        *g_data* is a void pointer in C but an integer in Python, so must be 
        typecast before use.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> bond = Bond()
        >>> gout = cvode.NVector([0.0])
        >>> g_data = ctypes.c_float(0.0)
        >>> bond.repol()(0, bond.model.y0, gout, ctypes.byref(g_data)), gout
        (0, [-82.420...])
        >>> g_data.value = -75
        >>> bond.repol()(0, bond.model.y0, gout, ctypes.byref(g_data)), gout
        (0, [-7.420199...])
        """
        def result(t, y, gout, g_data):
            """Set gout[0] to difference between V and target value."""
            gout[0] = self.yr.V - ctypes.cast(g_data, 
                ctypes.POINTER(ctypes.c_float)).contents.value
            return 0
        return result
    
    def _ap_without_rootfinding(self, p_repol=(0.25, 0.5, 0.75, 0.9), 
        ignore_flags=False):
        r"""
        Simulate action potential triggered by stimulus current.
        
        :param array_like y: initial state
        :param array_like p_repol: compute action potential duration to 
           :math:`p_{repol} \times 100\%` repolarization
        :param bool ignore_flags: disable sanity checking of the shape of the 
           action potential
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> cell = Bond()
        >>> t, y, stats = cell.ap(p_repol=(0.25, 0.5))
        >>> from pprint import pprint # platform-independent order of dict items
        >>> pprint(stats)
        {'amp': 115.44...,
         'base': -82.4201999...,
         'caistats': {'amp': 0.525...},
         'decayrate': array([ 0.223...
         'peak': 33.021572...,
         't_repol': array([  3.31012...,   5.125...]),
         'ttp': 1.844189...}
        """
        
        result = [] # list to keep results from integration subintervals

        # integrate over stimulus
        # logging.debug("integrate over stimulus")
        result.append(self.integrate(t=[0, self.pr.stim_duration], 
            nrtfn=0, assert_flag=cvode.CV_TSTOP_RETURN, 
            ignore_flags=ignore_flags))
        
        # integrate from the end of this stimulus to the start of the next one:
        # logging.debug("integrate from end of stimulus to start of next")
        # RHS discontinuity here, so force the solver to re-initialize: len(t)>1
        result.append(self.integrate(
            t=[self.pr.stim_duration, self.pr.stim_period], 
            nrtfn=0, assert_flag=cvode.CV_TSTOP_RETURN, 
            ignore_flags=ignore_flags))
        
        # logging.debug("integration finished")
        t, Y, _flag = zip(*result) # (t, Y, flag), where each is a tuple
        
        # The items of the tuples refer to these intervals
        # 0) stimulus
        # 1) stimulus to just before next stimulus
        
        # concatenation converts recarray to ndarray, so need to convert back
        Y = np.concatenate(Y).view(np.recarray)
        t = np.concatenate(t)
        # logging.debug("computing action potential duration")
        stats = ap_stats.apd(t, Y.V, p_repol=p_repol)
        assert t[stats["i"]][0] == stats["ttp"]
        try:
            sc = stats["caistats"] = ap_stats.apd(t, Y.Cai)
            # don't bother to report Ca decay rate for very small oscillations 
            if (sc["amp"] / sc["peak"]) < 1e-3:
                sc["decayrate"] = np.nan
        except AttributeError: # model may not have a state variable called Cai
            pass
        return t, Y, stats
    
    def aps(self, n=5, y=None, pr=None, *args, **kwargs):
        """
        Consecutive stimulus-induced action potentials using :meth:`ap`.
        
        :param n: number of action potentials to run
        :param array_like y: initial state
        :param array_like pr: parameter set, or a sequence of one parameter set 
          for each action potential, in which case ``n = len(pr)`` is used.
        :rtype: list of one (time, states, stats) tuple per action potential
        
        Further arguments are passed to :meth:`ap`.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> bond = Bond()
        >>> aps = list(bond.aps(n=2, p_repol=(0.25, 0.5)))
        
        You can iterate over the list of tuples like so:
        
        >>> from pprint import pprint
        >>> for t, y, stats in aps:
        ...     print t[-1], y[-1].V
        ...     pprint(stats)
        71.43 [-84.000...]
        {'amp': 115.44...,
         'caistats': {'amp': 0.525...},
         'decayrate': array([ 0.223...}
        142.86 [-84.072...]
        {'amp': 110.86...,
         'caistats': {'amp': 0.208...},
         'decayrate': array([ 0.193...}
        
        Separate lists for time, states, stats:
        
        >>> t, y, stats = zip(*aps)
        
        Time is reckoned consecutively, not restarting for each action potential.
        
        >>> t[0][-1] == t[1][0]
        True
        
        Parameters can vary between intervals by specifying *pr* as a list.
        
        >>> p = np.tile(bond.pr, 3)
        >>> p["stim_period"] = 20, 30, 40
        >>> with bond.autorestore():
        ...     [ti[-1] for ti, yi, statsi in bond.aps(pr=p)]
        [20.0, 50.0, 90.0]
        
        In case of a :exc:`~cvodeint.core.CvodeException`, the exception 
        instance has a *result* attribute with the results thus far: 
        A list of *(t, y, stats)*, where *stats* is *None* for the failed action 
        potential.
                
        (Doctests to guard against accidental changes.)
        
        >>> ix = np.r_[0:2, -2:0]
        >>> t[1][ix]
        array([  71.43      ,   71.43012...,  142...,  142.86      ])
        >>> y[1].V[ix]
        array([[-84.000...], [-83.99...], [-84.07...], [-84.07...]])
        """
        t0 = 0.0
        y0 = y
        if pr is None:
            pr = np.tile(self.pr, n)
        else:
            n = len(pr)
        
        for p in pr:
            with self.autorestore(_p=p, _y=y0):
                t, y, stats = self.ap(*args, **kwargs)
            t = t + t0
            yield t, y, stats
            t0, y0 = t[-1], y[-1]
    
    def converged(self, x0, x1, reltol={}, abstol={}):  # pylint: disable=W0102
        """
        Convergence checking of x0[k] vs x1[k] for k in reltol and abstol.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> bond = Bond()
        >>> with bond.autorestore():
        ...     x0, x1 = [np.trapz(y.view(float), t, axis=0).view(y.dtype) 
        ...         for t, y, stats in bond.aps(n=2)]
        >>> max([abs(j/i-1) for i, j in zip(x0.item(), x1.item())])
        2.05...
        >>> bond.converged(x0, x1, reltol=2.1)
        True
        >>> bond.converged(x0, x1, reltol=2)
        False
        
        >>> bond.converged(x0, x1)
        Traceback (most recent call last):
        AssertionError: Must specify relative or absolute tolerance
        """    
        # Note that the "reltol" we use to check convergence of action 
        # potentials is something else than the reltol used by the CVODE 
        # integrator.
        if np.isscalar(reltol):
            reltol = dict((k, reltol) for k in self.dtype.y.names)
        assert reltol or abstol, "Must specify relative or absolute tolerance"
        absconv = all([abs(x1[k] - x0[k]) < tol for k, tol in abstol.items()])
        relconv = all([abs((x1[k] - x0[k]) / x0[k]) < tol 
            for k, tol in reltol.items()])
        return relconv and absconv
        
    def steady(self, winwidth=10, max_nap=1000, reltol=0.001):
        """
        Run heart cell to approximate steady state.
        
        Return value is ((period, number of intervals to convergence), steady) 
        where steady is a list of *(t, y, stats, int_)* for the last *period* 
        intervals. *int_* contains the integral of each state variable's 
        trajectory.
        If state variables include *Cai*, the cycle is aligned so that the 
        highest Cai peak occurs in *steady[0]*.
        
        If dynamics does not converge within *max_nap* intervals, *period* is zero.
        
        To speed up the doctest, we use a precomputed approximate steady state.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> bond = Bond()
        >>> y0 = [-83.61, 0.30, 0.32, 703.76, 1167.91, 1.52e-02, 25.07, 134.76, 
        ...     1.16e-02, 1.64e-06, 0.79, 6.15e-10, 3.77e-04, 1.45e-04, 
        ...     9.67e-05, 6.85e-09, 1.02e-03, 0.14, 15493.40, 3.26e-07, 
        ...     1.49e-04, 1.27e-02, 7.86e-03, 1.27e-04, 1.91e-04, 1.56e-02, 
        ...     0.53, 141974.13, 2.36e-03, 1.00, 0.17, 0.69, 1.38e-02, 0.17, 
        ...     0.89, 0.90, 1.00, 4.79e-03, 9.07e-04, 1.91e-03, 8.43e-04]
        >>> with bond.autorestore(_y=y0):
        ...     (period, intervals), steady = bond.steady()
        >>> (t, y, stats, int_), = steady       # unpacking a 1-tuple
        >>> period, intervals, stats
        (1, 6, {'caistats': {'base': 0.298..., 'peak': 0.408...
        't_repol': array([ 31.0...,  40.2...}, 'base': -83.60..., 
        'peak': 25.35..., 't_repol': array([  3.7...,   5.7...})
        """
        d = deque(maxlen=winwidth)
        for i, (ti, yi, statsi) in enumerate(self.aps(n=max_nap)):
            # Converged if integral of specified state variables are 
            # within tolerance of values stored in deque, i.e. at t-1, t-2, ...
            # Trapezoidal integration of trajectories
            inti = np.trapz(yi.view(float), ti, axis=0).view(yi.dtype)
            for j, (_tj, yj, _statsj, intj) in enumerate(reversed(d)):
                if (self.converged(inti, intj, reltol) and 
                    self.converged(yi[0], yj[0], reltol)):
                    period = j + 1
                    steady = deque(d, maxlen=period)
                    if "Cai" in self.dtype.y.names:
                        # Ensure cycle starts with a high Cai peak
                        n = np.argmax([max(y.Cai) 
                            for _t, y, _stats, _int_ in steady])
                        if n: # Highest Cai peak is not yet in position 0
                            # Append the next interval, already computed
                            steady.append((ti, yi, statsi, inti))
                            # Compute the remaining n-1, if any
                            for tk, yk, statsk in self.aps(n=n-1, y=yi[-1]):
                                intk = np.trapz(yk.view(float), 
                                    tk, axis=0).view(yk.dtype)
                                steady.append((tk + ti[-1], yk, statsk, intk))
                    t0 = steady[0][0][0]
                    result = [(tk - t0, yk, statsk, intk) 
                        for tk, yk, statsk, intk in steady]
                    return (period, i), result
            # Don't append until after we've compared against previous items
            d.append((ti, yi, statsi, inti))
        t0 = d[0][0][0]
        result = [(tk - t0, yk, statsk, intk) for tk, yk, statsk, intk in d]
        # If we get here, convergence failed.
        return (0, i), result  # pylint: disable=W0631
    
    def restitution_portrait(self, BCL0=1000, delta=50, Delta=100, 
        tburnin=60000, nbetween=10, p_repol=0.70, *args, **kwargs):
        r"""
        Restitution portrait sensu :doi:`Kalb et al. 2004 <10.1046/j.1540-8167.2004.03550.x>`.
        
        All times are in milliseconds.
        
        :param BLC0: Initial basic cycle length
        :param delta: Change in cycle length for single-beat perturbations
        :param Delta: Change in cycle length between runs towards steady state
        :param tburnin: Burn-in period towards steady state
        :param nbetween: Number of stimuli separating the steps of the protocol
        :param p_repol: Use :math:`APD_{p\times100\%}` for action potential 
            duration
        
        Further arguments are passed to :meth:`ap_plain`.
        
        The protocol prescribes six steps that are repeated until 2:1 alternans 
        occurs. Default parameters take about five minutes.
        
        The return value is a list of lists of named tuples with fields: 
        
        * *step* : Roman numeral
        * *name* : name of step as defined by 
          :doi:`Kalb et al. 2004 <10.1046/j.1540-8167.2004.03550.x>`
        * *bcl* : basic cycle length
        * *description* : description of step
        * *R* : recovery curve; record array with names *di*, *apd* for 
          diastolic interval and action potential duration
        
        Model state and parameters are restored to their initial values 
        after running the protocol.
        
        Usage examples; not yet quality-controlled doctests.
        
        >>> from cgp.virtexp.elphys.examples import Tentusscher
        >>> cell = Tentusscher()
        >>> rp = cell.restitution_portrait(BCL0=150, tburnin=500, nbetween=5)
        >>> rp
        <generator object restitution_portrait at 0x...>
        >>> L = list(rp)
        >>> last_step = L[-1][-1]
        >>> last_step
        Step(step='VI', name='RCB-S', bcl=[50, 50, 50, 50, 50], 
        description='Recovery toward steady state', 
        R=rec.array([(28.00..., 23.70...), (26.29..., 24.72...),
                     (25.27..., 24.95...), (25.04..., 26.79...)], 
        dtype=[('di', '<f8'), ('apd', '<f8')]))
        >>> [[len(i.R) for i in j] for j in L]
        [[3, 5, 1, 5, 1, 4], [10, 5, 1, 5, 1, 4]]
        """
        Step = namedtuple("Step", "step name bcl description R")
        while True: # until a 2:1 response occurs
            # Perturbed downsweep pacing protocol, Kalb et al. p. 699
            nburnin = tburnin // BCL0
            pacingprotocol = [Step(*i, R=None) for i in [ 
            # Step  Name     Basic cycle length    Description
            ["I",   "RCB-D", [BCL0] * nburnin,  "Burn-in to steady state"], 
            ["II",  "R*",    [BCL0] * nbetween, "Measure at steady state"], 
            ["III", "RL",    [BCL0 + delta],    "Long coupling interval"], 
            ["IV",  "RCB-S", [BCL0] * nbetween, "Recovery toward steady state"],
            ["V",   "RS",    [BCL0 - delta],    "Short coupling interval"], 
            ["VI",  "RCB-S", [BCL0] * nbetween, "Recovery toward steady state"]
            ]]
            bcl = np.concatenate([step.bcl for step in pacingprotocol])
            pr = np.tile(self.pr, len(bcl))
            pr.stim_period = bcl
            with self.autorestore():
                aps = self.aps(pr=pr, p_repol=p_repol, *args, **kwargs)
                apd = [float(stats["t_repol"]) for _t, _y, stats in aps]
            di = pr.stim_period - apd
            R = np.rec.fromarrays([di[:-1], apd[1:]], names="di, apd")
            # split back into steps
            splits = np.cumsum([len(step.bcl) for step in pacingprotocol])
            RL = np.array_split(R, splits[:-1])
            yld = [step._replace(R=rl)  # pylint: disable=W0212
                   for step, rl in zip(pacingprotocol, RL)]
             
            # # Fit dynamic restitution curve
            # # APD = a - b exp(-DI/tau) for all R*
            # # ln(APD-a) = ln b - (1/tau) DI
            # from cgp.utils.rnumpy import r
            # Rs, = [step.R for step in yld if step.name == "R*"]
            # a, b = np.linalg.lstsq(
            #     np.c_[np.ones_like(Rs.di), Rs.di], Rs.apd)[0]
            # start = dict(a=mean(Rstar.apd), b=0, tau=1)
            # r.nls("apd~a-b*exp(-di/tau)", rec2dict(Rstar), start=start)

            yield yld
            BCL0 = BCL0 - Delta
            # Check for "2:1 response", whatever that means
            if BCL0 <= self.pr.stim_duration:
                break # while

# Convert between local time, starting at 0 in each interval, and global time.

def globaltime(T):
    """
    Return cumulative "global" time from list of time vectors starting at 0

    >>> T = [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]
    >>> globaltime(T)
    [[0, 1, 2], [2, 3, 4, 5], [5, 6, 7, 8, 9]]

    If you prefer a single vector:
    
    >>> np.concatenate(globaltime(T))
    array([0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9])
    """
    offset = np.cumsum([i[-1] for i in T])
    offset = np.r_[0, offset[:-1]]
    return [[o + ti for ti in t] for o, t in zip(offset, T)]

def localtime(T):
    """
    Return "local" time from a list of consecutive time vectors

    >>> T = [1, 2], [3, 4, 5], [6, 7, 8, 9]
    >>> localtime(T)
    [array([0, 1]), array([0, 1, 2]), array([0, 1, 2, 3])]
    """
    return [np.asanyarray(t) - t[0] for t in T]

def ap_stats_array(stats):
    """
    Convert action potential statistics from dict to structured ndarray.
    
    >>> from numpy import array
    >>> stats = {'amp' : 0.5, 'base': array([1.0]), 
    ...     'caistats': {'amp': 1.5, 'base': 2.0,
    ...         'decayrate': array([2.5]), 'i': array([1, 2, 3, 4, 5]), 
    ...         'p_repol': array([0.25, 0.5, 0.75, 0.9 ]),
    ...         'peak': 3.0, 't_repol': array([4.0, 5.0, 6.0, 7.0]), 'ttp': 8.0},
    ...     'decayrate': array([9.0]), 'i': array([6, 7, 8, 9, 10]),
    ...     'p_repol': array([0.25, 0.5, 0.75, 0.9]), 'peak': array([10.0]),
    ...     't_repol': array([11.0, 12.0, 13.0, 14.0]), 'ttp': 15.0}
    >>> ap_stats_array(stats)
    rec.array([ (0.5, 1.0, 10.0, 15.0, 9.0, 11.0, 12.0, 13.0, 14.0, 1.5, 
    2.0, 3.0, 8.0, 2.5, 4.0, 5.0, 6.0, 7.0)], 
    dtype=[('apamp', '<f8'), ('apbase', '<f8'), ('appeak', '<f8'), 
    ('apttp', '<f8'), ('apdecayrate', '<f8'), ('apd25', '<f8'), 
    ('apd50', '<f8'), ('apd75', '<f8'), ('apd90', '<f8'), ('ctamp', '<f8'), 
    ('ctbase', '<f8'), ('ctpeak', '<f8'), ('ctttp', '<f8'), 
    ('ctdecayrate', '<f8'), ('ctd25', '<f8'), ('ctd50', '<f8'), 
    ('ctd75', '<f8'), ('ctd90', '<f8')])
    
    >>> from cgp.virtexp.elphys.examples import Tentusscher
    >>> tt = Tentusscher()
    >>> t, y, stats = tt.ap()
    >>> ap_stats_array(stats)
    rec.array([ (121.10..., -86.2..., 34.90..., 1.35..., 0.0183..., 220.0..., 
    298.3..., 321.9..., 330.1..., 0.00050..., 0.0002..., 0.00070..., 10.2..., 
    0.0158..., 40.42..., 74.47..., 122.7..., 167.3...)], 
    dtype=[('apamp', '<f8'), ('apbase', '<f8'), ('appeak', '<f8'), 
    ('apttp', '<f8'), ('apdecayrate', '<f8'), ('apd25', '<f8'), 
    ('apd50', '<f8'), ('apd75', '<f8'), ('apd90', '<f8'), ('ctamp', '<f8'), 
    ('ctbase', '<f8'), ('ctpeak', '<f8'), ('ctttp', '<f8'), 
    ('ctdecayrate', '<f8'), ('ctd25', '<f8'), ('ctd50', '<f8'), 
    ('ctd75', '<f8'), ('ctd90', '<f8')])
    """
    names = "amp base peak ttp decayrate".split()
    n = len(names)
    names += ["d%d" % (100 * i) for i in stats["p_repol"]]
    if "caistats" in stats:
        dtype = [(var + name, float) for var in "ap", "ct" for name in names]
        data = np.r_[[float(stats[k]) for k in names[:n]],
                     stats["t_repol"],
                     [float(stats["caistats"][k]) for k in names[:n]],
                     stats["caistats"]["t_repol"]]
    else:
        dtype = [("ap" + name, float) for name in names]
        data = np.r_[[float(stats[k]) for k in names[:n]], stats["t_repol"]]
    return np.rec.array(data, dtype=dtype)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS | 
                    doctest.NORMALIZE_WHITESPACE)
