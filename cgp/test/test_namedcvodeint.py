"""Tests for :mod:`cgp.cvodeint.namedcvodeint`."""

from nose.tools import raises
import numpy as np

from ..cvodeint.namedcvodeint import Namedcvodeint

@raises(Exception)
def test_autorestore():
    """Require correct size for state vector."""
    n = Namedcvodeint()
    with n.autorestore(_y=(1, 2, 3)):
        pass

def test_no_p():
    """Test with an ODE without parameters."""
    def ode(_t, y, ydot, _g_data):
        """ODE right-hand side."""
        ydot[:] = -y
    
    n = Namedcvodeint(ode, t=[0, 1], y=np.ones(1.0).view([("y", float)]))
    with n.autorestore():
        n.integrate()
