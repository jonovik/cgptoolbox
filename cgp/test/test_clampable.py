"""Tests for `cgp.virtexp.elphys.clampable`."""

from cgp.virtexp.elphys import clampable

def test_pairbcast():
    """Work around 32-D limit on rank of ndarray.ndim."""
    protocol = [((1, 2), 3)] + [(1, 1)] * 15 + [(4, (5, 6))]
    clampable.pairbcast(*protocol)
