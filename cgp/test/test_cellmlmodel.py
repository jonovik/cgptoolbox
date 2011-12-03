"""Tests for :mod:`cgp.physmod.cellmlmodel`."""
# pylint: disable=C0111, E0611, F0401, E1101

import numpy as np
from nose.tools import assert_equal

from ..physmod.cellmlmodel import Cellmlmodel, Legend, parse_legend

vdp = Cellmlmodel()
vdp_compiled = Cellmlmodel(use_cython=True)
vdp_uncompiled = Cellmlmodel(use_cython=False)

def test_rates_and_algebraic():
    exposure_workspace = ("b0b1820b1376263e16c6086ca64d513e/"
                          "bondarenko_szigeti_bett_kim_rasmusson_2004_apical")
    for use_cython in False, True:
        bond = Cellmlmodel(exposure_workspace, t=[0, 5], 
                           use_cython=use_cython, reltol=1e-5)
        bond.yr.V = 100 # simulate stimulus
        t, y, _flag = bond.integrate()
        ydot, alg = bond.rates_and_algebraic(t, y)
        actual = ydot.V[-1], ydot.Cai[-1], alg.i_Na[-1]
        desired = [[-4.08092831], [ 0.06698888], [-1.70527191]]
        np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)
    
def test_parse_legend():
    """Protect against empty legend entry, bug in CellML code generation."""
    assert_equal(parse_legend(["x in component A (u)", ""]), 
        Legend(name=('x', 'empty__1'), component=('A', ''), unit=('u', '')))

def test_compiled_behaviour():
    assert str(vdp_uncompiled.model).startswith(
        "<module 'cgp.physmod.cellml2py.vanderpol_vandermark_19285756af26cf")
    assert str(vdp_compiled.model).startswith(
        "<module 'cgp.physmod.cellml2py.cython.vanderpol")
    # This module holds the ode() function passed to CVode.
    assert str(vdp_uncompiled.model.ode).startswith("<function ode at")
    # If compiled, it appears as a built-in function.
    assert str(vdp_compiled.model.ode) == "<built-in function ode>"

def test_source():
    """Alert if code generation changes format."""
    import hashlib
    assert_equal(hashlib.sha1(vdp.py_code).hexdigest(), 
        '6236832ecb45498c4cab07ea7c214371cd5fa1fe')

def test_Sundials_convention():    
    """
    Verify that the generated ODE conforms to the Sundials convention.
     
    Return 0 for success, <0 for unrecoverable error (and perhaps >0 for 
    recoverable error, i.e. if a shorter step size might help).
    """    
    # Any exception raised in an uncompiled ODE is silenced, 
    # and the function returns -1.
    # Here, the input to my_f_ode() is invalid, but no exception is raised.
    assert_equal(-1, vdp_uncompiled.my_f_ode(None, None, None, None))

def test_properties():    
    """
    Parameters and initial value arrays for a model module.
    
    Note that the default initial state is a module-level variable shared by 
    all instances of this model object. Subsequent calls to Cellmlmodel() will 
    see that the model is already imported and not redo the initialization.
    """
    vdp = Cellmlmodel()  # pylint: disable=W0621
    assert_equal(1, vdp.model.p)
    np.testing.assert_equal([-2, 0], vdp.model.y0)
    # Updates by name reflect in the original array.
    # (This returns an array; use x[0] if you really need a scalar.)
    assert_equal(-2, vdp.y0r.x)
    try:
        vdp.y0r.x = 3.14
        np.testing.assert_equal([3.14, 0], vdp.model.y0)
    finally:
        vdp.y0r.x = -2.0  # undo change
    
