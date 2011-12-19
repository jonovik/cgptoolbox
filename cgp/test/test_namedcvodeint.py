from nose.tools import raises

from ..cvodeint.namedcvodeint import Namedcvodeint

@raises(Exception)
def test_autorestore():
    """Require correct size for state vector."""
    n = Namedcvodeint()
    with n.autorestore(_y=(1, 2, 3)):
        pass
