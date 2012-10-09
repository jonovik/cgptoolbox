"""Test :mod:`cgp.virtexp.elphys.examples`."""
# pylint: disable=E0611

from nose.tools import assert_equal, assert_not_equal

from ..virtexp.elphys import *
from ..virtexp.elphys.examples import *

def test_hodgkin():
    """
    Test moved from :mod:`cgp.virtexp.elphys.examples`.
    
    >>> hh = Hodgkin()
    >>> t, y, stats = hh.ap()
    >>> ap_stats_array(stats)
    rec.array([ (107.3..., -75.0, 32.3..., 
    2.20..., 0.993..., 3.241..., 
    4.013..., 4.843..., 5.269...)],
    dtype=[('apamp', '<f8'), ('apbase', '<f8'), ('appeak', '<f8'), 
    ('apttp', '<f8'), ('apdecayrate', '<f8'), ('apd25', '<f8'), 
    ('apd50', '<f8'), ('apd75', '<f8'), ('apd90', '<f8')])
    >>> [stats["peak"] for t, y, stats in hh.aps(n=2)]
    [32.5688..., 32.5687...]
    
    Verify fix for 0/0 bug.
    
    >>> with hh.autorestore(V=-50):
    ...     t, y, stats = hh.ap()
    
    Before the fix, this produced:
    Traceback (most recent call last):
    CvodeException: CVode returned CV_CONV_FAILURE
    """
    pass

def test_scenario():
    """
    Test .scenario() feature.
    
    >>> bond = Bond()
    >>> with bond.scenario("septal"):
    ...     t0, y0, stats0 = bond.ap()
    
    The two scenarios are available as separate models at cellml.org.
    Verify that switching scenarios is equivalent to using the other version.
    
    >>> septal = Bond(workspace=bond.workspace, exposure=bond.exposure, 
    ...     changeset=bond.changeset, 
    ...     variant=bond.variant.replace("apical", "septal"))
    >>> with septal.autorestore():
    ...     t1, y1, stats1 = septal.ap()
    
    Compare string representations to allow for machine imprecision.
    
    >>> str(ap_stats_array(stats0)) == str(ap_stats_array(stats1))
    True
    
    >>> with septal.scenario("apical"):
    ...     t2, y2, stats2 = septal.ap()
    >>> with bond.autorestore():
    ...     t3, y3, stats3 = bond.ap()
    >>> str(ap_stats_array(stats2)) == str(ap_stats_array(stats3))
    True
    """
    pass

def test_fitz_paced():
    """
    Test "paced" scenario of Fitz model.
    
    The constructor defines a "paced" :meth:`~Bond.scenario` where small 
    periodic stimuli elicit action potentials. To hack this, we impose a 
    negative stimulus most of the time, removing it briefly to elicit the 
    action potential.
    
    >>> fitz = Fitz()
    >>> with fitz.scenario("paced"):
    ...     L = list(fitz.aps(n=5))
    >>> [stats["ttp"] for t, y, stats in L[-2:]] # last two action potentials
    [1.0535..., 1.0535...]
    """
    pass

def test_li():
    """
    Test :class:`Li`.
    
    >>> li = Li()
    >>> t, y, stats = li.ap(rootfinding=True)
    >>> from pprint import pprint
    >>> pprint(stats)
    {'amp': array([ 115.42...,
     'base': array([-78.9452...
     'peak': array([ 36.48...,
     't_repol': array([  4.285...,   5.699...,  10.216...,  17.28...]),
     'ttp': 3.14...}
    """
    pass

def test_bond_uhc():
    """
    Compare original and unhardcoded :class:`Bond` model.
    
    Verify that action potential and calcium transient statistics are equal 
    for the original and unhardcoded model, to within roundoff error.
    
    >>> b = Bond()
    >>> bu = Bond_uhc()
    >>> t, y, stats = b.ap(rootfinding=True)
    >>> tu, yu, statsu = bu.ap(rootfinding=True)
    >>> a, au = [ap_stats_array(i) for i in stats, statsu]
    
    >>> for k in a.dtype.names:
    ...     try:
    ...         np.testing.assert_almost_equal(a[k], au[k], decimal=3)
    ...     except AssertionError:
    ...         print k, a[k], au[k]
    ctttp [ 15.68...] [ 15.69...]
    """
    pass

def test_li_uhc():
    """
    Compare original and unhardcoded :class:`Li` model.
    
    >>> li = Li()
    >>> liu = Li_uhc()
    >>> t, y, stats = li.ap(rootfinding=True)
    >>> tu, yu, statsu = liu.ap(rootfinding=True)
    >>> a, au = [ap_stats_array(i) for i in stats, statsu]
    >>> for k in a.dtype.names:
    ...     tol = 1.5e-3 if (k == "ctttp") else 1e-3
    ...     if abs(1 - a[k] / au[k]) > tol:
    ...         print k, a[k], au[k]
    """
    pass

from ..virtexp.elphys.examples import Bond

def test_set_y_before_ap():
    """Test setting of state before simulating action potential."""
    bond = Bond()
    with bond.autorestore():
        for V0 in bond.y0r.V + [-5, 0, 5]:
            bond.y[:] = bond.model.y0
            bond.yr.V = V0
            _t, y, _stats = bond.ap()
            assert_equal(y.V[0], V0)

def test_not_overwrite_y0r():
    """Check that bond.y0r is not overwritten when the model is integrated."""
    bond = Bond()
    with bond.autorestore():
        bond.y[:] = bond.model.y0
        _t, y, _stats = bond.ap()
        assert_not_equal(y.V[-1], bond.y0r.V)

def test_tentusscher():
    """
    Test for class :class:`cgp.virtexp.elphys.Tentusscher`.
    
    >>> import numpy as np
    >>> tt = Tentusscher()
    >>> t, y, stats = tt.ap()    
    >>> from pprint import pprint
    >>> pprint(stats)
    {'amp': 121.10...,
     'base': -86.2...,
     'caistats': {'amp': 0.0005032...,
                  'base': 0.0002000...,
                  'decayrate': array([ 0.0158...]),
                  'i': array([...]),
                  'p_repol': array([ 0.25,  0.5 ,  0.75,  0.9 ]),
                  'peak': 0.0007032...,
                  't_repol': array([  40.42...,   74.47...,  122.7...,  167.3...]),
                  'ttp': 10...},
     'decayrate': array([ 0.0183...]),
     'i': array([...]),
     'p_repol': array([ 0.25,  0.5 ,  0.75,  0.9 ]),
     'peak': 34.90...,
     't_repol': array([ 220.03...,  298.33...,  321.90...,  330.140...]),
     'ttp': 1.3...}
    
    Alert if numerical results change.
    
    >>> [float(np.trapz(y[k], t, axis=0)) for k in "V", "Cai"]
    [-56243..., 0.1418...]
    """
    pass
