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
"""

import numpy as np
from pysundials import cvode

from ..physmod.cellmlmodel import Cellmlmodel
from . import ap_stats

class Paceable(object):
    """
    Mixin class for pacing electrophysiological models.
    
    This protocol is applicable to any model with a transmembrane potential 
    *V* and stimulus parameters *stim_amplitude* and *stim_period*.
    
    To use, define a class inheriting both from :class:`Paceable` and 
    :class:`~cgp.physmod.cellmlmodel.Cellmlmodel`. See the source code for 
    class :class:`Bond` for an example, including defining the model 
    identifier as the default *exposure_workspace* for the class constructor.
    """
    
    def ap(self, y=None, pr=None, p_repol=(0.25, 0.5, 0.75, 0.9), 
        ignore_flags=False):
        """
        Simulate action potential triggered by stimulus current.
        
        :param array_like y: initial state
        :param array_like p_repol: compute action potential duration to 
           :math:`p_{repol} \times 100\%` repolarization
        :param bool ignore_flags: disable sanity checking of the shape of the 
           action potential
        
        >>> cell = Bond()
        >>> t, y, stats = cell.ap()
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
        
        >>> with cell.autorestore(V=-60, stim_period=50):
        ...     t, y, stats = cell.ap()
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
        result.append(self.integrate(t=[0, self.pr.stim_duration], y=y, 
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

class Bond(Cellmlmodel, Paceable):
    """
    Example for class :class:`Paceable` using the Bondarenko et al. 2004 model.
    
    Please **see the source code** for how this class uses the 
    :class:`Paceable` mixin to add an experimental protocol to a 
    :class:`~cgp.physmod.cellmlmodels.Cellmlmodel`.
    
    ..  inheritance-diagram: cgp.physmod.cellmlmodel.Cellmlmodel Paceable Bond
        parts: 1
    
    .. todo:: Add voltage clamping.
    
    As a convenience feature, the :meth:`Bond` constructor defines the 
    *exposure_workspace* model identifier as a default argument.
    Another typical adjustment is to set the redundant parameter *stim_start* 
    to zero.
    
    Once defined, the :class:`Bond` class can be used as follows:
    
    ..  plot::
        :include-source:
        :width: 300
        
        from cgp.virtexp.elphys import Bond
        bond = Bond()
        t, y, stats = bond.ap()
        plt.plot(t, y.V)
    
    References:
    
    * :doi:`original paper <10.1152/ajpheart.00185.2003>`
    * :cellml:`CellML implementation 
      <11df840d0150d34c9716cd4cbdd164c8/bondarenko_szigeti_bett_kim_rasmusson_2004_apical>`
    """
    
    _exposure_workspace = ("11df840d0150d34c9716cd4cbdd164c8/" +
        "bondarenko_szigeti_bett_kim_rasmusson_2004_apical")
    
    def __init__(self, exposure_workspace=_exposure_workspace, 
                 *args, **kwargs):
        """Constructor for the :class:`Bond` class."""
        super(Bond, self).__init__(exposure_workspace, *args, **kwargs)
        if "stim_start" in self.dtype.p.names:
            self.pr.stim_start = 0.0
