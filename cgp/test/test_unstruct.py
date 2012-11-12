"""Tests for :mod:`cgp.utils.unstruct`."""

import numpy as np
from cgp.utils.unstruct import unstruct

def test_dimensions():
    """Check relationship between dimensions of x and unstruct(x)."""
    # x has 2 fields and dimensions (3, 4)
    x = np.arange(24).view(dtype=[(i, int) for i in "ab"]).reshape(3, 4)
    # u has dimensions (3, 4, 2)
    u = unstruct(x)
    # change to u affects x
    u[:] += 10
    # Last dimension of u corresponds to fields of x
    np.testing.assert_equal(x["a"], u[..., 0])
    # First dimensions of u corresponds to first dimensions of x
    np.testing.assert_equal(x[0][0].item(), u[0][0])

def test_dtypes():
    """Verify that unstruct() handles different dtypes correctly."""
    for fieldtype in np.int8, np.int32, float:
        dtype = [(i, fieldtype) for i in "ab"]
        x = np.arange(4, dtype=fieldtype).view(dtype)
        yield np.testing.assert_equal, unstruct(x), [[0, 1], [2, 3]]

def test_zero_rank():
    """
    Handle intricacies of zero-rank arrays.
    
    `Zero-rank arrays 
    <http://projects.scipy.org/numpy/wiki/ZeroRankArray>`_ 
    are tricky; they can be structured, yet be of type numpy.void.

    Here, x0 and x1 are almost, but not completely, the same:
    
    >>> fieldtype = np.int32   # ensure same result on 32- and 64-bit platforms
    >>> dtype = [("a", fieldtype), ("b", fieldtype)]
    >>> x0 = np.array([(0, 1)], dtype=dtype)[0]
    >>> x1 = np.array((0, 1), dtype=dtype)

    Despite a lot of equalities below, x0 and x1 are of different type.
    
    >>> (x0 == x1) and (x0.shape == x1.shape) and (x0.dtype == x1.dtype)
    True
    >>> x0
    (0, 1)
    >>> x1
    array((0, 1), dtype=[('a', '<i4'), ('b', '<i4')])
    >>> type(x0), type(x1)
    (<type 'numpy.void'>, <type 'numpy.ndarray'>)
    
    Unstructuring them was tricky, but finally works.
    
    >>> unstruct(x0)
    array([0, 1]...)
    >>> unstruct(x1)
    array([0, 1]...)
    """
    fieldtype = np.int32   # ensure same result on 32- and 64-bit platforms
    dtype = [("a", fieldtype), ("b", fieldtype)]
    x0 = np.array([(0, 1)], dtype=dtype)[0]
    x1 = np.array((0, 1), dtype=dtype)
    np.testing.assert_equal(unstruct(x0), unstruct(x1))
