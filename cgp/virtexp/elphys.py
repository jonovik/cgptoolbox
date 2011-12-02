"""
Virtual experiments for cellular electrophysiology.

Assumed interface:

* :data:`V` : Transmembrane voltage
* :data:`i_stim` : Stimulus current
"""

class Paceable(object):
    """
    Mixin class for pacing electrophysiological models.
    
    Applicable to any model with a transmembrane potential and stimulus 
    current.
    
    To use, define a class inheriting both from :class:`Paceable` and 
    :class:`~cellmlmodels.cellmlmodel.Cellmlmodel`. You may wish to specify 
    the model identifier (*exposure_workspace*) as a default argument to the 
    constructor.
    
    >>> exposure_workspace = "11df840d0150d34c9716cd4cbdd164c8/"
            "bondarenko_szigeti_bett_kim_rasmusson_2004_apical"
    >>> class Bond(Namedcvodeint, Paceable):
    ...     pass
    >>> bond = Bond(exposure_workspace)
    >>> t, y, stats = bond.ap()
    >>> plt.plot(t, y.V)
    """
    
        def ap_plain(self, y=None, pr=None, p_repol=r_[0.25, 0.5, 0.75, 0.9], ignore_flags=False):
        """
        Simulate action potential triggered by stimulus current. No rootfinding.
        
        Arguments are as for :meth:`ap`.
        
        >>> bond = Bond()
        >>> t, y, stats = bond.ap_plain()
        >>> from pprint import pprint # platform-independent order of dict items
        >>> pprint(stats)
        {'amp': 115.44...,
         'base': -82.4201999...,
         'caistats': {'amp': 0.525...},
         'decayrate': array([ 0.0911...
         'peak': 33.021572...,
         't_repol': array([  3.31012...,   5.125...,  14.282...,  22.78...]),
         'ttp': 1.844189...}
        
        The arguments *y* and *pr* are largely obsolete; use 
        :meth:`~cvodeint.namedcvodeint.Namedcvodeint.autorestore` 
        instead to temporarily modify initial state or parameters.
        
        >>> with bond.autorestore(V=-60, stim_period=50):
        ...     t, y, stats = bond.ap_plain()
        >>> t[-1]
        50.0
        >>> pprint(stats)
        {'amp': 87.75...,
         'base': -60.0,
         'caistats': {'amp': 0.208...},
         'decayrate': array([ 0.195...
         'peak': 27.75...,
         't_repol': array([  2.562...,   3.667...,   7.152...,  11.882...]),
         'ttp': 1.342...}
        """
        
        if y is None:
            y = self.y
        if pr is not None:
            self.pr[:] = pr
        self._ReInit_if_required(t=0, y=y)
        
        result = [] # list to keep results from integration subintervals

        # integrate over stimulus
        # logging.debug("integrate over stimulus")
        result.append(self.integrate(t=[0, self.pr.stim_duration], y=y, nrtfn=0,  
            assert_flag=cvode.CV_TSTOP_RETURN, ignore_flags=ignore_flags))
        
        # integrate from the end of this stimulus to the start of the next one:
        # logging.debug("integrate from end of stimulus to start of next")
        # RHS discontinuity here, so force the solver to re-initialize: len(t)>1
        result.append(self.integrate(t=[self.pr.stim_duration, self.pr.stim_period], nrtfn=0, 
            assert_flag=cvode.CV_TSTOP_RETURN, ignore_flags=ignore_flags))
        
        # logging.debug("integration finished")
        t, Y, flag = zip(*result) # (t, Y, flag), where each is a tuple
        
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
