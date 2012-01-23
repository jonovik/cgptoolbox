"""Tests for :mod:`cgp.virtexp.elphys.clampable`."""

import numpy as np
# http://readthedocs.org/docs/nose/en/latest/plugins/attrib.html
from nose.plugins.attrib import attr

from cgp.virtexp.elphys.clampable import catrec, mmfits

# FIXME: --attr='!slow' --with-doctests will prevent normal tests from running
# For the time being, I rename this file to _test... so it won't run.

@attr(slow=True)
def test_vargap():
    """Test fast vs slow variable-gap protocol."""
    from cgp.virtexp.elphys.examples import Bond
    b = Bond()
    protocol = [(100, -80), (50, 0), 
                (np.arange(2, 78, 15), (-90, -80, -70)), (100, 0)]
    p1, gap, p2 = b.vargap(protocol)  # 4 s
    np.testing.assert_equal([len(i) for i in p1, gap, p2], [1, 3, 18])
    L = b.vecvclamp(protocol)  # 9 s
    np.testing.assert_equal(
        [len(i) for i in zip(*[traj for _proto, traj in L])], [18, 18, 18, 18])

@attr(slow=True)
def test_bond_protocols(plot=False):
    """Run all Bondarenko protocols."""
    
    from cgp.virtexp.elphys.examples import Bond
    b = Bond(chunksize=10000)
    
    def plotprot(i, varnames, limits, L):
        """Plot one protocol."""
        from pylab import figure, subplot, plot, savefig, legend, axis
        figure()
        proto0, _ = L[0]
        isclamped = len(proto0[0]) == 2
        for j, (n, lim) in enumerate(zip(varnames, limits)):
            subplot(len(varnames), 1, j + 1)
            leg = set()
            for k in n.split():
                if isclamped: # clamp protocol
                    # For each protocol, traj is a list with a Trajectory
                    # for each pulse.
                    for _proto, traj in L:
                        t, y, _dy, a = catrec(*traj[1:])
                        plot(t, y[k] if k in y.dtype.names else a[k], 
                            label=k if (k not in leg) else None)
                        leg.add(k)
                else: # pacing protocol
                    # For each protocol, paces is a list of Pace
                    for _proto, paces in L:
                        t, y, _dy, a, _stats = catrec(*paces)
                        plot(t, y[k] if k in y.dtype.names else a[k], 
                            label=k if (k not in leg) else None)
                        leg.add(k)
                legend()
            if lim:
                axis(lim)
        savefig("fig%s%s.png" % (i, b.name))
    
    bp = b.bond_protocols()
    for i, (varnames, protocol, limits, _url) in bp.items():
        if len(protocol[0]) == 2: # (duration, voltage), so clamping
            # List of (proto, traj), where traj is list of Trajectory
            L = b.vecvclamp(protocol)
        else: # (n, period, duration, amplitude), so pacing
            # List of (proto, paces), where paces is list of Pace
            L = b.vecpace(protocol)
        if plot:
            plotprot(i, varnames, limits, L)

def test_mmfits():
    """
    Michaelis-Menten fit of peak i_CaL current vs gap duration.
    
    This uses a reduced version of the variable-gap protocol of Bondarenko's 
    figure 7. The full version is available as b.bond_protocols()[7].protocol.
    """
    from cgp.virtexp.elphys.examples import Bond
    b = Bond()
    protocol = (1000, -80), (250, 0), (np.linspace(2, 202, 5), -80), (100, 0)
    with b.autorestore():
        L = b.vecvclamp(protocol)
    np.testing.assert_allclose(mmfits(L, k="i_CaL"), (6.44, 18.14), rtol=1e-3)
    # Verify improvement of error message in case k is list rather than string.
    try:
        mmfits(L, 2, ["i_Na"])
    except AssertionError, exc:
        msg = "k must be a single field name of y or a, not <type 'list'>"
        assert msg in str(exc)
