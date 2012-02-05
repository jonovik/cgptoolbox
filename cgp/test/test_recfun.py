"""Tests for :mod:`cgp.utils.recfun`."""
# pylint: disable=C

import numpy as np
from nose.tools import raises

from cgp.utils import recfun 

a = np.array([(0, 1)], dtype=[("a", float), ("b", float)])
b = np.array([(2, 3)], dtype=[("c", float), ("d", float)])

@raises(TypeError)
def test_cbind_empty():
    recfun.cbind()

def test_cbind_void():
    """Check that ()-shaped arrays are OK."""
    recfun.cbind(a[0], b[0])

def test_cbind_squeeze():
    """Check that trailing singleton dimensions are squeezed."""
    recfun.cbind(np.tile(a, 2), np.tile(b, 2).reshape(2, 1))
