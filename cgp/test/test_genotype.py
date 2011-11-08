"""Tests for :mod:`cgp.gt.genotype`."""
# pylint: disable=C0111

import numpy as np
from numpy.testing import assert_equal
from nose.tools import raises
from ..gt.genotype import Genotype

def test_genotype_init():
    assert_equal(np.array(Genotype([3, 2])), 
              [[1, 1],
               [1, 0],
               [0, 1],
               [0, 0],
               [2, 1],
               [2, 0]])

@raises(AssertionError)
def test_genotype_biallelic():
    Genotype([4, 4])
