"""Tests for :mod:`cgp.physmod.cellmlmodel`."""
# pylint: disable=C0111
from ..physmod.cellmlmodel import Cellmlmodel

def test_cellmlmodel_init():
    Cellmlmodel()

def test_rates_and_algebraic():
    exposure_workspace = "b0b1820b1376263e16c6086ca64d513e/bondarenko_szigeti_bett_kim_rasmusson_2004_apical"
    for use_cython in False, True:
        bond = Cellmlmodel(exposure_workspace, t=[0, 5], use_cython=use_cython)
        bond.yr.V = 100 # simulate stimulus
        t, y, _flag = bond.integrate()
        _ydot, _alg = bond.rates_and_algebraic(t, y)
