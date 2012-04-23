"""Tests for :mod:`cgp.utils.hdfcache`."""
# pylint: disable=W0603, C0111, W0612, W0212

import os
import tempfile

import numpy as np

from cgp.utils.hdfcache import Hdfcache

dtemp = None

def setup():
    global dtemp
    dtemp = tempfile.mkdtemp()

def teardown():
    import shutil
    try:
        shutil.rmtree(dtemp)
    except OSError:
        pass

def test_hdfcache():
    filename = os.path.join(dtemp, 'cachetest.h5')
    hdfcache = Hdfcache(filename)

    @hdfcache.cache
    def f(x, a, b=10):
        "This is the docstring for function f"
        result = np.zeros(1, dtype=[("y", float)])
        result["y"] = x["i"] * 2 + x["f"] + a * b
        print "Evaluating f:", x, a, b, "=>", result
        return result
    
    desired = "@hdfcache.cache\ndef f(x, a, b=10):"
    actual = "\n".join(line.strip() 
        for line in hdfcache.file.root.f._v_attrs.sourcecode.split("\n"))
    np.testing.assert_string_equal(actual[:len(desired)], desired)
    
    desired = __file__.replace(".pyc", ".py")
    actual = hdfcache.file.root.f._v_attrs.sourcefile
    np.testing.assert_string_equal(actual[-len(desired):], desired)
