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
from contextlib import contextmanager
from collections import namedtuple, deque
import ctypes

import numpy as np
from pysundials import cvode

from ...physmod.cellmlmodel import Cellmlmodel
from . import Paceable, Clampable, ap_stats


class Bond(Cellmlmodel, Paceable, Clampable):
    """
    :mod:`cgp.virtexp.elphys` example: Bondarenko et al. 2004 model.
    
    Please **see the source code** for how this class uses the 
    :class:`Paceable` and :class:`Clampable` mixins to add an experimental 
    protocols to a :class:`~cgp.physmod.cgp.physmod.Cellmlmodel`.
    
    ..  inheritance-diagram: cgp.physmod.cellmlmodel.Cellmlmodel Paceable Clampable Bond
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


class Tentusscher(Cellmlmodel, Paceable):
    """
    Example for class :class:`Paceable` - Ten Tusscher heart M-cell model.

    Reference: :doi:`Ten Tusscher et al. 2004 <10.1152/ajpheart.00794.2003>`.
  
    .. plot::
       :width: 300
       
       from cgp.virtexp.elphys import Tentusscher
       tt = Tentusscher()
       t, y, stats = tt.ap()
       fig = plt.figure()
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
        exposure_workspace="c7f7ced1e002d9f0af1b56b15a873736/"
            + "tentusscher_noble_noble_panfilov_2004_a",
        rename={"y": {"Na_i": "Nai", "Ca_i": "Cai", "K_i": "Ki"}, "p": {
            "IstimStart": "stim_start", 
            "IstimEnd": "stim_end", 
            "IstimAmplitude": "stim_amplitude", 
            "IstimPeriod": "stim_period", 
            "IstimPulseDuration": "stim_duration"
        }}, **kwargs):
        kwargs["rename"] = rename
        super(Tentusscher, self).__init__(exposure_workspace, **kwargs)
        self.pr.stim_start = 0

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
    
    >>> tt = Tentusscher()
    >>> t, y, stats = tt.ap()
    >>> ap_stats_array(stats)
    rec.array([ (121.10..., -86.2..., 34.90..., 1.35..., 0.050..., 220.0..., 
    298.3..., 321.9..., 330.1..., 0.00050..., 0.0002..., 0.00070..., 10.27..., 
    0.017..., 40.42..., 74.47..., 122.7..., 167.3...)], 
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
