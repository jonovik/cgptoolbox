"""Tests for :mod:`..cvodeint.core`."""

import numpy as np
from nose.tools import raises
from pysundials import cvode

from ..cvodeint import *

def test_ReInit():
    """
    Check reinitialization depending on the form of t and y.

    >>> t, y = [0, 2], [0.5]
    >>> o = Cvodeint(example_ode.exp_growth, t, y)
    >>> for y in [None, [0.25]]:
    ...     for t in [None, 1, [1], [1, 2], [1, 2, 3]]:
    ...         o._ReInit_if_required(t, y)
    ...         print y, o.y, t, o.t, o.t0, o.tstop
    None [0.5] None [0 2] c_double(0.0) 2
    None [0.5] 1 [0.0, 1] c_double(0.0) 1
    None [0.5] [1] [0.0, 1] c_double(0.0) 1
    None [0.5] [1, 2] [1 2] c_double(1.0) 2
    None [0.5] [1, 2, 3] [1 2 3] c_double(1.0) 3
    [0.25] [0.25] None [1 2 3] c_double(1.0) 3
    [0.25] [0.25] 1 [1.0, 1] c_double(1.0) 1
    [0.25] [0.25] [1] [1.0, 1] c_double(1.0) 1
    [0.25] [0.25] [1, 2] [1 2] c_double(1.0) 2
    [0.25] [0.25] [1, 2, 3] [1 2 3] c_double(1.0) 3

    If y is a Numpy structured or record array, its .item() method is used 
    to get a plain tuple. Testing this both with and without t specified.
    
    >>> yrec = np.rec.fromarrays([[1]], names="fieldname")
    >>> o._ReInit_if_required(t=None, y=yrec)
    >>> print y, o.y, t, o.t, o.t0, o.tstop
    [0.25] [1.0] [1, 2, 3] [1 2 3] c_double(1.0) 3
    >>> o._ReInit_if_required(t=2, y=yrec)
    >>> print y, o.y, t, o.t, o.t0, o.tstop
    [0.25] [1.0] [1, 2, 3] [1.0, 2] c_double(1.0) 2
    """
    pass

def test_integrate_adaptive_steps():
    """
    Verify bugfix for UnboundLocalError.
    
    At some point, integrating to some time <= the current time would cause
    "UnboundLocalError: local variable 'flag' referenced before assignment".
    """
    c = Cvodeint(example_ode.logistic_growth, t=[0, 2], y=[0.1], reltol=1e-3)
    c.integrate()
    t, y, flag = c.integrate(t=1)
    np.testing.assert_equal(flag, cvode.CV_TSTOP_RETURN)

@raises(CvodeException)
def test_maxsteps():
    """
    Verify that the maxsteps argument is honored.
    """
    c = Cvodeint(example_ode.logistic_growth, t=[0, 2], y=[0.1], maxsteps=3)
    c.integrate()

def test_newchunk():
    """Test the adding of new chunks of memory."""
    c = Cvodeint(example_ode.logistic_growth, t=[0, 2], y=[0.1], chunksize=30)
    c.integrate()

def test_repr():
    """Test string representation of Cvodeint object."""
    from doctest import _ellipsis_match # comparison with ... ellipsis
    got = repr(Cvodeint(example_ode.exp_growth, t=[0,2], y=[0.1]))
    wants = ["Cvodeint(f_ode=exp_growth, t=array([0, 2]), y=[0.1...], " +
             "abstol=c_double(1e-08))",
            "<class 'cvodeint.Cvodeint'> <function ode at 0x..." + 
            "getargspec(__init__) is not available when running under Cython"]
    if not any([_ellipsis_match(want, got) for want in wants]):
        print "Wanted:", wants, "\\n", "Got:", got

def test_logging():
    """Produces error message and traceback that are ignored by doctest."""
    cvodeint = Cvodeint(example_ode.logging_ode, [0, 4], [1, 1])
