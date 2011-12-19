"""Test :mod:`cgp.virtexp.ap_cvode`."""

import numpy as np
from nose.tools import nottest

from ..virtexp.ap_cvode import Bond

@nottest  # comment out to make test_stim_amplitude() fail
def test_this():
    """This interferes with the test below."""
    from cgp.physmod.cellmlmodel import Cellmlmodel
    bond = Cellmlmodel("11df840d0150d34c9716cd4cbdd164c8/"
        "bondarenko_szigeti_bett_kim_rasmusson_2004_apical")
    with bond.dynclamp(-140) as clamped:
        pass

def test_stim_amplitude():
    """Debug unexpected zero stim_amplitude."""
    bond = Bond()
    _t, _y, _stats = bond.ap()
    # Stimulation starts at time 0.0, overriding any default in the CellML
    np.testing.assert_equal(bond.pr.stim_start, 0)
    # Parameters governing the stimulus setup
    L = [(s, bond.pr[s]) for s in bond.pr.dtype.names if s.startswith("stim_")]
    np.testing.assert_equal(L, 
        [('stim_start', 0), 
         ('stim_end', 100000),
         ('stim_period', 71.43),
         ('stim_duration', 0.5),
         ('stim_amplitude', -80)])
