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
"""  # pylint: disable=C0301,E1002,W0611

from __future__ import division # 7 / 4 = 1.75 rather than 1

import numpy as np

from ...physmod.cellmlmodel import Cellmlmodel
from . import Paceable, Clampable
from ...utils.ordereddict import OrderedDict


class Hodgkin(Cellmlmodel, Paceable, Clampable):
    """
    Hodgkin-Huxley model of action potential.
    
    I have made some hacks to make this model more commensurable with the 
    Bondarenko (2004) model. The current 
    :cellml:`CellML version <5d116522c3b43ccaeb87a1ed10139016/hodgkin_huxley_1952>` 
    has stimulus duration and amplitude hardcoded, and the stimulus is not 
    periodic. I have hacked this in the CellML source for now.
    I've also fixed a 0/0 bug in alpha_m for V == -50.
    
    ..  plot::
    
        from cgp.virtexp.elphys.examples import Hodgkin
        hh = Hodgkin()
        t, y, stats = hh.ap()
        plt.plot(t, y.V)
    """
    def __init__(self, localfile="hodgkin_huxley_1952", **kwargs):
        super(Hodgkin, self).__init__(localfile=localfile, **kwargs)

class Tentusscher(Cellmlmodel, Paceable, Clampable):
    """
    Example for class :class:`Paceable` - Ten Tusscher heart M-cell model.

    Reference: :doi:`Ten Tusscher et al. 2004 <10.1152/ajpheart.00794.2003>`.
  
    .. plot::
       :width: 300
       
       from cgp.virtexp.elphys.examples import Tentusscher
       tt = Tentusscher()
       t, y, stats = tt.ap()
       plt.plot(t, y.V, '.-')
       i = stats["i"]
       plt.plot(t[i], y.V[i], 'ro')
    
    This model's `main CellML page
    <http://models.cellml.org/exposure/c7f7ced1e002d9f0af1b56b15a873736>`_
    links to three versions of the model corresponding to different cell types:   
    :cellml:`a (midmyocardial)
    <c7f7ced1e002d9f0af1b56b15a873736/tentusscher_noble_noble_panfilov_2004_a>`,
    :cellml:`b (epicardial)
    <c7f7ced1e002d9f0af1b56b15a873736/tentusscher_noble_noble_panfilov_2004_b>`,
    :cellml:`c (endocardial)
    <c7f7ced1e002d9f0af1b56b15a873736/tentusscher_noble_noble_panfilov_2004_c>`.
    
    Other keyword arguments are passed through to 
    :meth:`~cgp.physmod.cellmlmodel.Cellmlmodel`.
    In particular, the "rename" argument changes some state and parameter 
    names to follow the conventions of a :class:`Paceable` object.
    """
    def __init__(self,  # pylint: disable=W0102
        workspace="tentusscher_noble_noble_panfilov_2004",
        rename={"y": {"Na_i": "Nai", "Ca_i": "Cai", "K_i": "Ki"}, "p": {
            "IstimStart": "stim_start", 
            "IstimEnd": "stim_end", 
            "IstimAmplitude": "stim_amplitude", 
            "IstimPeriod": "stim_period", 
            "IstimPulseDuration": "stim_duration"
        }}, **kwargs):
        kwargs["rename"] = rename
        super(Tentusscher, self).__init__(workspace=workspace, **kwargs)
        self.pr.stim_start = 0

class Bond(Cellmlmodel, Paceable, Clampable):
    """
    :mod:`cgp.virtexp.elphys` example: Bondarenko et al. 2004 model.
    
    Please **see the source code** for how this class uses the 
    :class:`Paceable` and :class:`Clampable` mixins to add an experimental 
    protocols to a :class:`~cgp.physmod.cgp.physmod.Cellmlmodel`.
    
    ..  inheritance-diagram:: cgp.physmod.cellmlmodel.Cellmlmodel Paceable Clampable Bond
        :parts: 1
    
    .. todo:: Add voltage clamping.
    
    Note: The redundant parameter *stim_start* is set to zero.
    
    Once defined, the :class:`Bond` class can be used as follows:
    
    ..  plot::
        :include-source:
        :width: 300
        
        from cgp.virtexp.elphys.examples import Bond
        bond = Bond()
        t, y, stats = bond.ap()
        plt.plot(t, y.V)
        
    References:
    
    * :doi:`original paper <10.1152/ajpheart.00185.2003>`
    * :cellml:`CellML implementation 
      <11df840d0150d34c9716cd4cbdd164c8/bondarenko_szigeti_bett_kim_rasmusson_2004_apical>`
    """
    
    def __init__(self, workspace="bondarenko_szigeti_bett_kim_rasmusson_2004", 
                 *args, **kwargs):
        """Constructor for the :class:`Bond` class."""
        super(Bond, self).__init__(workspace=workspace, *args, **kwargs)
        if "stim_start" in self.dtype.p.names:
            self.pr.stim_start = 0.0
        # Mapping None to an empty dict, and letting the scenario name default 
        # to None, makes self.scenario() equivalent to self.autorestore().
        self.scenarios = OrderedDict({None: {}})
        kwargs = kwargs.copy()
        kwargs.update(workspace=self.workspace, exposure=self.exposure)
        for variant in self.get_variants():
            name = variant.replace(self.workspace, "").replace("_", "")
            self.scenarios[name] = dict(workspace=self.workspace, 
                exposure=self.exposure, changeset=self.changeset, 
                variant=variant, y=self.y0r.copy(), p=self.pr.copy())
            kwargs["variant"] = variant
            m = Cellmlmodel(**kwargs)
            m.pr.stim_start = 0.0
            for k in set(m.dtype.p.names) & set(self.dtype.p.names):
                self.scenarios[name]["p"][k] = m.pr[k]
            for k in set(m.dtype.y.names) & set(self.dtype.y.names):
                self.scenarios[name]["y"][k] = m.y0r[k]
    
    def scenario(self, name=None, **kwargs):
        """
        Context manager to set parameters and initial state to a named scenario.
        
        This is just a wrapper for 
        :meth:`~cvodeint.namedcvodeint.Namedcvodeint.autorestore`, 
        and defaults to a plain :meth:`autorestore` if *name* = None. 
        A Bond object has scenarios representing "apex" and "septum" cells, 
        with apex as the default. The two scenarios are available as separate 
        models at cellml.org.
        Subclasses may define additional scenarios. Note that scenarios will 
        need adaptation to work with subclasses that have different parameter 
        names.
        
        ``**kwargs`` are passed to 
        :meth:`~cvodeint.namedcvodeint.Namedcvodeint.autorestore`.
        
        ..  plot::
        
            from cgp.virtexp.elphys.examples import Bond
            bond = Bond()
            for k in "apical", "septal":
                with bond.scenario(k):
                    t, y, stats = bond.ap()
                plt.plot(t, y.V, label=k)
                plt.legend()
        
        >>> bond = Bond()
        >>> bond.scenarios
        OrderedDict([(None, {}), 
        ('apical', {'changeset': '99f4fd6804311c571a7143515003691ab2e430fb', 
            'workspace': 'bondarenko_szigeti_bett_kim_rasmusson_2004', 
            'p': rec.array([ (1.0, ...)], dtype=[('Cm', '<f8'), ...)]), 
            'y': rec.array([ (-82.4202, ...)], dtype=[('V', '<f8'), ...]), 
            'variant': 'bondarenko_szigeti_bett_kim_rasmusson_2004_apical', 
            'exposure': '11df840d0150d34c9716cd4cbdd164c8'}), 
        ('septal', {'changeset': '99f4fd6804311c571a7143515003691ab2e430fb', 
            'workspace': 'bondarenko_szigeti_bett_kim_rasmusson_2004', 
            'p': rec.array([ (1.0, ...)], dtype=[('Cm', '<f8'), ...)]), 
            'y': rec.array([ (-82.4202, ...)], dtype=[('V', '<f8'), ...]), 
            'variant': 'bondarenko_szigeti_bett_kim_rasmusson_2004_septal', 
            'exposure': '11df840d0150d34c9716cd4cbdd164c8'})])
        """
        return self.autorestore(_y=self.scenarios[name].get("y"), 
                                _p=self.scenarios[name].get("p"), **kwargs)

class Fitz(Bond):
    r"""
    CellML implementation of the FitzHugh-Nagumo nerve axon model.
    
    References:
        
    * Nagumo, J., Animoto, S., Yoshizawa, S. (1962)
      :doi:`An active pulse transmission line simulating nerve axon 
      <10.1109/JRPROC.1962.288235>`.
      Proc. Inst. Radio Engineers, 50, 2061-2070.
    * FitzHugh R (1961) 
      :doi:`Impulses and physiological states in theoretical models of nerve 
      membrane <10.1016/S0006-3495(61)86902-6>`. 
      Biophysical J. 1:445-466
    
    In Fitzhugh (1961), the definition of the transmembrane potential is 
    such that it decreases during depolarization, so that the action potential 
    starts with a downstroke, contrary to the convention used in FitzHugh 1969 
    and in most other work. The equations are also somewhat rearranged. 
    However, figure 1 of FitzHugh 1961 gives a very good overview of the phase 
    plane of the model.
    
    The nullclines of the model are:
    
    .. math::
    
        \dot{v} = 0 &\Leftrightarrow& w = v (v-\alpha) (1-v) + I \\
        \dot{w} = 0 &\Leftrightarrow& w = (1/\gamma) v
    
    A high gamma makes w change slowly relative to v, making the system more 
    stiff and the action potentials more "square".
    In the absence of a stimulus current, the model has a limit cycle if 
    :math:`\alpha \gamma < -1`
    
    I have made some hacks to make this model more commensurable with the 
    Bondarenko (2004) model. The current 
    :cellml:`CellML version <cf32346a9e5c4b2cdb559b11da5f1ae1/fitzhugh_1961>` 
    has stimulus duration and amplitude hardcoded, and the stimulus is not 
    periodic. I have hacked this in the CellML source for now.
    
    Also, I rename the state variable *v* to *V* for compatibility with the 
    pacing and clamping protocols.
    
    The hardcoded stimulus protocol in the CellML version is strange in that 
    the stimulus *decreases* the transmembrane potential, and with a magnitude 
    far beyond that of the model's action potential. My guess is that 
    *stim_duration* and *stim_amplitude* were copied directly from the 
    Bondarenko model.
    
    .. plot::
       :include-source:
       :width: 300
       
       from cgp.virtexp.elphys.examples import Fitz
       fitz = Fitz(reltol=1e-10)
       for t, y, stats in fitz.aps(n=3):
           plt.plot(t, y.view(float))
    
    >>> fitz = Fitz(reltol=1e-10)
    >>> with fitz.autorestore():
    ...     t, y, stats = fitz.ap()
    >>> [float(stats[k]) for k in "base peak ttp".split()]
    [0.0, 0.984..., 28.2...]
    
    In fact, this parameter scenario is self-exciting even without a stimulus 
    current. :math:`(V=0, w=0)` is an equilibrium (though unstable), so we 
    choose a different initial value.
    
    .. plot::
        :include-source:
        :width: 300
        
        from cgp.virtexp.elphys.examples import Fitz
        fitz = Fitz()
        with fitz.autorestore(stim_amplitude=0, V=0.01):
            t, y, flag = fitz.integrate(t=[0, 700])
        plt.plot(t, y.view(float))
        plt.legend(fitz.dtype.y.names)
    
    The constructor defines a "paced" :meth:`~Bond.scenario` where small 
    periodic stimuli elicit action potentials. To hack this, we impose a 
    negative stimulus most of the time, removing it briefly to elicit the 
    action potential.
    """
        
    def __init__(self, localfile="fitzhugh_1961",  # pylint: disable=W0102
            rename={"y": {"v": "V"}}, **kwargs):
        """
        Return a Fitzhugh (1961) model object.
        
        Keyword arguments are passed through to 
        :class:`~cellmlmodels.cellmlmodel.Cellmlmodel`.
        
        In particular, the *rename* argument changes some state and parameter 
        names to match those of the Bondarenko model, from which this class is 
        derived.
        
        See the class docstring for examples.
        """
        kwargs["rename"] = rename
        super(Fitz, self).__init__(localfile=localfile, **kwargs)
        pr = self.pr.copy()
        pr.stim_period = 200
        pr.stim_duration = 190
        pr.stim_amplitude = -0.1
        self.scenarios["paced"] = dict(p=pr, y=(0.01, 0.01))

class Li(Cellmlmodel, Paceable, Clampable):
    """
    CellML implementation of the LNCS modified Bondarenko model by Li et al.
    
    .. seealso:: :doi:`10.1152/ajpheart.00219.2010`
    """
    
    def __init__(self, localfile="BL6WT_260710", **kwargs):
        """
        Return a Li-Niederer-Casadei-Smith (LNCS) model object.
        
        "exposure_workspace" will eventually refer to the cellml.org repository.
        Currently, I work with code generated from the CellML file using 
        OpenCell 0.8.
        Other keyword arguments are passed through to 
        :mod:`cellmlmodels.cellmlmodel.Cellmlmodel`.
        
        This constructor sets *stim_offset* = 0.0, overriding the default. 
        This is because the stepwise integration assumes that the stimulus is 
        at the start of the action potential.
        """
        super(Li, self).__init__(localfile=localfile, **kwargs)
        # Assume AP starts with stimulus
        self.pr.stim_offset = 0.0
        # Disable caffeine injection unless specifically requested
        if "prepulses_number" in self.pr.dtype.names:
            self.pr.prepulses_number = np.inf

class Bond_uhc(Bond):
    """
    Bondarenko model with most constants unhardcoded.
    
    Comparing the details of the original and unhardcoded Bond models.
    
    >>> b = Bond(reltol=1e-10)
    >>> bu = Bond_uhc(reltol=1e-10)
    
    Increased number of parameters.
    
    >>> len(b.dtype.p), len(bu.dtype.p)
    (73, 204)
    
    Verify that the original parameters still have the same value.
    
    >>> for k in np.intersect1d(b.dtype.p.names, bu.dtype.p.names):
    ...     if b.pr[k] != bu.pr[k]:
    ...         print k, b.pr[k], bu.pr[k]
    
    The variable iKss was dropped because it was really a constant.
    
    >>> list(np.setdiff1d(b.dtype.y.names, bu.dtype.y.names))
    ['iKss']
    
    The sodium-calcium exchanger current was renamed.
    
    >>> list(np.setxor1d(b.dtype.a.names, bu.dtype.a.names))
    ['i_NCX', 'i_NaCa']
    """
    def __init__(self, localfile="bond_uhc", *args, **kwargs):
        super(Bond_uhc, self).__init__(localfile=localfile, *args, **kwargs)

class Li_uhc(Li):
    """
    LNCS model with most constants unhardcoded.
    
    >>> li = Li()
    >>> liu = Li_uhc()
    >>> len(li.dtype.p), len(liu.dtype.p)
    (86, 188)
    >>> for k in np.intersect1d(li.dtype.p.names, liu.dtype.p.names):
    ...     if li.pr[k] != liu.pr[k]:
    ...         print k, li.pr[k], liu.pr[k]
    >>> list(np.setdiff1d(li.dtype.y.names, liu.dtype.y.names)) # dropped variable
    ['iKss']
    >>> li.dtype.a == liu.dtype.a
    True
    """
    def __init__(self, localfile="li_uhc", *args, **kwargs):
        super(Li_uhc, self).__init__(localfile=localfile, *args, **kwargs)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS | 
                    doctest.NORMALIZE_WHITESPACE)
