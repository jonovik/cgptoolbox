"""Tests for :mod:`cgp.phenotyping.attractor`."""
# pylint: disable=C0111, W0612, W0613, E1101, R

import numpy as np
from nose.tools import raises

from .. import cvodeint
from ..phenotyping import attractor

class Test(cvodeint.Cvodeint, attractor.AttractorMixin):  # pylint:disable=W
    pass

def ode(t, y, ydot, g_data):
    ydot[:] = -y

def test_exponential_decay():
    r"""
    :math:`y(t) = e^{-t}`, so that :math:`|y|=tol \Leftrightarrow t=\ln tol`.
    """
    test = Test(ode, t=[0, 1500], y=[1])
    tol = 1e-6
    t, y = test.eq(tol=tol)
    np.testing.assert_approx_equal(y, tol)
    np.testing.assert_approx_equal(t, -np.log(tol), significant=4)
    
def test_logistic_growth():
    test = Test(cvodeint.example_ode.logistic_growth, t=[0, 100], y=0.1)
    t, y, _flag = test.eq(tol=1e-8, last_only=False)
    np.testing.assert_allclose(y[-1], 1)
    np.testing.assert_allclose(t[-1], 20.687, rtol=1e-4)

def test_already_converged():
    test = Test(ode, t=[0, 1e5], y=0)
    t, _y, _flag = test.eq(last_only=False)
    assert len(t) == 1

@raises(cvodeint.core.CvodeException)
def test_not_converged():
    test = Test(ode, t=[0, 1], y=[1])
    test.eq()