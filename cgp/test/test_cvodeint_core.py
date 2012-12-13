"""Tests for :mod:`..cvodeint.core`."""

from doctest import _ellipsis_match # comparison with ... ellipsis

import numpy as np
from nose.tools import raises
from pysundials import cvode

from ..cvodeint import (  # pylint: disable=F0401
    Cvodeint, CvodeException, example_ode)
import pickle

def test_CvodeException():
    """
    Test :exc:`cgp.cvodeint.CvodeException`.
    
    >>> from ..cvodeint import cvodefun
    >>> i = 5     # countdown
    >>> @cvodefun
    ... def failsoon(t, y, ydot, f_data):
    ...     global i
    ...     if i:
    ...         ydot[0] = i    # trick solver into many evaluations
    ...         i -= 1
    ...     else:
    ...         raise StandardError("Testing CvodeException")
    >>> cvodeint = Cvodeint(failsoon, [0, 1], [1])
    >>> try:
    ...     cvodeint.integrate()
    ... except CvodeException, exc:
    ...     print exc.result
    ...     print exc
    ...     print failsoon.traceback
    (array([ 0.]), array([[ 1.]]), -8)
    CVode returned CV_RHSFUNC_FAIL
    Traceback (most recent call last):
    ...
    StandardError: Testing CvodeException
    
    In addition, CVODE will print this message outside of standard output/error::
    
        [CVODE ERROR]  CVode
          At t = 0, the right-hand side routine failed in an unrecoverable manner.
    """
    pass

def test_ReInit():
    """
    Check reinitialization depending on the form of t and y.

    >>> t, y = [0, 2], [0.5]
    >>> o = Cvodeint(example_ode.exp_growth, t, y)
    >>> for y in [None, [0.25]]:
    ...     for t in [None, 1, [1], [1, 2], [1, 2, 3]]:
    ...         o._ReInit_if_required(t, y)
    ...         print y, o.y, t, o.t, o.t0, o.tstop
    None [0.5] None [ 0.  2.] c_double(0.0) 2.0
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
    _t, _y, flag = c.integrate(t=1)
    np.testing.assert_equal(flag, cvode.CV_TSTOP_RETURN)

def test_integrate():
    """Verify that integration works correctly."""
    c = Cvodeint(example_ode.logistic_growth, t=[0, 2], y=[0.1])
    t, y, _flag = c.integrate()
    ys = example_ode.logistic_growth_sol(t, [0.1])
    np.testing.assert_allclose(y.squeeze(), ys, rtol=1e-6)

def test_integrate_end():
    """Verify integration to tstop if ode changes at tstop."""
    c = Cvodeint(example_ode.nonsmooth_growth, t=[0, 1], y=[1], reltol=1e-10)
    t, y, _flag = c.integrate()
    ys = np.exp(t)
    np.testing.assert_allclose(y.squeeze(), ys)

def test_integrate_nonsmooth():
    """Verify that nonsmooth equation gets integrated correctly."""
    c = Cvodeint(example_ode.nonsmooth_growth, t=[0, 2], y=[1], reltol=1e-10)
    t, y, _flag = c.integrate()
    ys = example_ode.nonsmooth_growth_sol(t, y[0])
    np.testing.assert_allclose(y.squeeze(), ys, rtol=1e-6)

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
    got = repr(Cvodeint(example_ode.exp_growth, t=[0, 2], y=[0.1]))
    wants = ["Cvodeint(f_ode=exp_growth, t=array([0, 2]), y=[0.1...], " +
             "abstol=c_double(1e-08))",
            "<class 'cvodeint.Cvodeint'> <function ode at 0x..." + 
            "getargspec(__init__) is not available when running under Cython"]
    if not any([_ellipsis_match(want, got) for want in wants]):
        print "Wanted:", wants, "\\n", "Got:", got

import logging
from cStringIO import StringIO
import math
fmtstr = "%(" + ")s\t%(".join(
    "asctime levelname name lineno process message".split()) + ")s"

def test_logging():    
    """CVODE allows handling/logging of exceptions that occur in the ODE."""
    
    sio = StringIO()
    logger = logging.getLogger("test_logging")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sio)
    handler.setFormatter(logging.Formatter(fmtstr))
    logger.addHandler(handler)

    def logging_ode(t, y, ydot, f_data):
        """An ODE with logging as a side effect."""
        logger.debug("Evaluating ODE...")
        try:
            with np.errstate(divide="ignore"):
                ydot[0] = math.log(y[1])
                ydot[1] = 1 / y[0] - t # will eventually turn negative
        except StandardError: # allow KeyboardInterrupt, etc., to work
            logger.exception("Caught an exception when evaluating ODE.")
            return -1
        logger.debug("ODE evaluated without error.")
        return 0
    
    def assert_ellipsis_match(want):
        """Match like doctest +ELLIPSIS, tolerating "..."."""
        got = sio.getvalue().strip()
        if not _ellipsis_match(want, got):
            msg = "Logging output differs from the expected.\n"
            msg += "Wanted:\n{}\n\nGot:\n{}".format(want, got)
            raise AssertionError(msg)
    
    # Evaluate the ODE at a valid state
    t = np.array([0.0])
    y = np.array([0.0, 1.0])
    ydot = np.zeros_like(y)
    f_data = None
    want = "20...\tDEBUG\ttest_logging\t...\tEvaluating ODE...\n"
    want += "20...\tDEBUG\ttest_logging\t...\tODE evaluated without error."
    assert 0 == logging_ode(t, y, ydot, f_data)
    assert_ellipsis_match(want)
    # Reset the logger
    sio.reset()
    sio.truncate()
    # Evaluate the ODE at an invalid state
    y[1] = -1.0
    want = "20...\tDEBUG\ttest_logging\t...\tEvaluating ODE...\n"
    want += "20...\tERROR\ttest_logging\t...\t"
    want += "Caught an exception when evaluating ODE.\n"
    want += "Traceback (most recent call last):\n...\n"
    want += "ValueError: math domain error"
    assert -1 == logging_ode(t, y, ydot, f_data)
    assert_ellipsis_match(want)

def test_simple_example():
    """
    Simple example.
    
    >>> from ..cvodeint.example_ode import exp_growth
    >>> cvodeint = Cvodeint(exp_growth, t=[0,2], y=[0.1])
    >>> cvodeint.integrate()
        (array([  0.00000000e+00,   2.34520788e-04,   ... 2.00000000e+00]),
        array([[ 0.1       ], [ 0.10002346], ... [ 0.73890686]]), 1)
    """
    pass

def test_y_dtype():
    """Verify that y can be float or int (coerced to float)."""
    Cvodeint(example_ode.vdp, t=[0, 2], y=[1.0, 2.0])
    Cvodeint(example_ode.vdp, t=[0, 2], y=[1, 2])

@raises(ValueError, IndexError)
def test_y_dtype_rec():
    """Verify that y can be a record array (coerced to float)."""
    Cvodeint(example_ode.vdp, t=[0, 2], y=np.rec.fromrecords([(1.0, 2.0)], 
        dtype=[("u", float), ("v", float)]))

def test_pickling():
    """Verify that Cvodeint objects can be serialized."""
    old = Cvodeint(example_ode.logistic_growth, t=[0, 2], y=[0.1], reltol=1e-3)
    s = pickle.dumps(old)
    new = pickle.loads(s)
    for desired, actual in zip(old.integrate(), new.integrate()):
        np.testing.assert_array_equal(desired, actual)
    
