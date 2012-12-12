"""Tests for :mod:`cgp.physmod.cellmlmodel`."""
# pylint: disable=C0111, E0611, F0401, E1101

import hashlib

import numpy as np
from nose.tools import assert_equal

from ..physmod import cellmlmodel
from ..physmod.cellmlmodel import Cellmlmodel, Legend, parse_legend

vdp = Cellmlmodel()
vdp_compiled = Cellmlmodel(use_cython=True)
vdp_uncompiled = Cellmlmodel(use_cython=False)

def test_purge():
    Cellmlmodel(purge=True)

def test_hash():
    a, b, c = [Cellmlmodel(chunksize=i) for i in 1, 1, 2]
    assert hash(a) == hash(b)
    assert hash(b) != hash(c)

def test_autorestore_reinit():
    """Guard against bug setting t=[0, 0] on autorestore."""
    c = Cellmlmodel()
    old = np.copy(c.t)
    with c.autorestore():
        np.testing.assert_equal(c.t, old)

def test_rates_and_algebraic():
    desired = [-3.67839219], [ 0.06717483], [-2.21334685]
    for use_cython in False, True:
        bond = Cellmlmodel(
            workspace="bondarenko_szigeti_bett_kim_rasmusson_2004", 
            t=[0, 5], use_cython=use_cython, reltol=1e-5)
        with bond.autorestore():
            bond.yr.V = 100 # simulate stimulus
            t, y, _flag = bond.integrate()
        ydot, alg = bond.rates_and_algebraic(t, y)
        actual = ydot.V[-1], ydot.Cai[-1], alg.i_Na[-1]
        np.testing.assert_allclose(actual, desired, rtol=1e-4, atol=1e-4)
    
def test_parse_legend():
    """Protect against empty legend entry, bug in CellML code generation."""
    assert_equal(parse_legend(["x in component A (u)", ""]), 
        Legend(name=('x', 'empty__1'), component=('A', ''), unit=('u', '')))

def test_compiled_behaviour():
    # This module holds the ode() function passed to CVode.
    assert str(vdp_uncompiled.model.ode).startswith("<function ode at")
    # If compiled, it appears as a built-in function.
    assert str(vdp_compiled.model.ode) == "<built-in function ode>"

def test_source():
    """Alert if code generation changes format."""
    assert_equal(hashlib.sha1(vdp.py_code).hexdigest(),
        '24bceba4077b7fcf53bf0198b32ffaf47585bb29')

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

def test_get_all_workspaces():
    w = cellmlmodel.get_all_workspaces()
    assert "A Primer on Modular Mass Action Modelling with CellML" in w.title
