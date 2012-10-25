"""Tests for `cgp.virtexp.elphys.clampable`."""

import numpy as np

from cgp.virtexp.elphys import clampable

def test_pairbcast():
    """Check that broadcasting works for long sequences of pairs."""
    pairs = [((1, 2), 3)] + [(1, 1)] * 15 + [(4, (5, 6))]
    # Broadcast all pairs
    bc = clampable.pairbcast(*pairs)
    # Broadcast only the varying pairs for comparison
    bc_ends = clampable.pairbcast(*(pairs[:1] + pairs[-1:]))
    # Check that the varying pairs match
    for i, j in zip(bc, bc_ends):
        np.testing.assert_equal(i[:1] + i[-1:], j)
    # Check that the constant pairs match
    for i in bc:
        for j in i[1:-1]:
            np.testing.assert_equal(j, (1, 1))
