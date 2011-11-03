"""Tests for :mod:`utils.placevalue`."""

import unittest

from numpy.testing import assert_equal, assert_raises

from ..utils.placevalue import Placevalue

p = Placevalue([4, 3, 2])

def test_init_properties():
    """Verify properties of Placevalue object."""
    assert_equal(p.n, [4, 3, 2])
    assert_equal(p.maxint, 24)
    assert_equal(p.posval, [6, 2, 1])
    assert_equal(Placevalue([4, 3, 2], msd_first=False).posval, [1, 4, 12])

def test_vec2int_digitrange():
    """Check range of digits."""
    assert_raises(OverflowError, p.vec2int, [0, 0, 2])

def test_int2vec_2d():
    """
    Check that int2vec(sequence) is 2-d even when there is only one position.
    
    Verify bugfix: If i is a sequence, the output should be a 2-d array 
    so that "for v in pv.int2vec(i)" works. Previously, this failed when 
    there was only one position.
    """
    a = Placevalue([3])
    assert_equal(a.int2vec(range(4)), [[0], [1], [2], [3]])

@unittest.skip("vec2int() overflows if maxint > range of fixed-size integers")
def test_vec2int_huge():
    a = Placevalue([[2] * 64])
    v = [0] + [1] * 63
    i = a.vec2int(v)
    # Overflows to -1
    assert_equal(i, 2**64-1)
    # Overflows to array([-1, -1,  1,  1,  1,  ...
    assert_equal(a.int2vec(i)[:5], [0, 1, 1, 1, 1])
