"""
Convert recarray to ordered dictionary.

This module was motivated by the need to export recarrays to R via rnumpy.
Using an OrderedDict preserves the order of lists.
"""

from collections import OrderedDict

import numpy as np

def rec2dict(x):
    """
    Convert recarray (or structured ndarray) to dictionary.
    
    See below for passing (nested) record arrays to R.
    
    The doctests below use ellipses, "...", to allow for minor 
    differences in output formatting.
    
    Basic usage:
    
    >>> import numpy as np
    >>> A = np.rec.fromrecords([(10, 1.5), (20, 2.5)], names='id, x')
    >>> A
    rec.array([(10, 1.5), (20, 2.5)], dtype=[('id', '<i...'), ('x', '<f8')])
    >>> rec2dict(A)
    OrderedDict([('id', array([10, 20])), ('x', array([ 1.5,  2.5]))])
    
    This works even with nested record arrays:
    
    >>> nested_dtype = [('id', '<i8'), ('x', '<f8'),
    ...     ('rec', [('a', '<i8'), ('b', '<f8')])]
    >>> B = np.rec.fromrecords([(10, 1.5, (1, 2)), (20, 2.5, (3, 4))],
    ...     dtype=nested_dtype)    
    >>> B
    rec.array([(10..., 1.5, (1..., 2.0)), (20..., 2.5, (3..., 4.0))],
          dtype=[('id', '<i8'), ('x', '<f8'), 
                 ('rec', [('a', '<i8'), ('b', '<f8')])])
    >>> rec2dict(B)
    OrderedDict([('id', array([10, 20]...)), 
                 ('x', array([ 1.5,  2.5])), 
                 ('rec', OrderedDict([('a', array([1, 3]...)), 
                                      ('b', array([ 2.,  4.]))]))])
    
    Passing a record array (possibly nested) to R:
    
    >>> from cgp.utils.rnumpy import *                                    # doctest: +SKIP
    >>> rcopy(rec2dict(A))                                      # doctest: +SKIP
    $x
    [1] 1 2
    $id
    [1] 10 20
    >>> r.as_data_frame(rec2dict(A))                            # doctest: +SKIP
      x id
    1 1 10
    2 2 20
    
    A nested record array becomes a nested list: 
    
    >>> rcopy(rec2dict(B))                                      # doctest: +SKIP
    $x
    [1] 1.5 2.5
    $id
    [1] 10 20
    $rec
    $rec$a
    [1] 1 3
    $rec$b
    
    Coercion to data frame auto-generates readable column names:
    
    >>> r.as_data_frame(rec2dict(B))                            # doctest: +SKIP
        x id rec.a rec.b
    1 1.5 10     1     2
    2 2.5 20     3     4
    """
    if x.dtype.names: # recarray or structured ndarray
        return OrderedDict((k, rec2dict(x[k])) for k in x.dtype.names)
    else:
        return x

def dict2rec(*args, **kwargs):
    """
    Convert dict of arrays to nested recarray.
    
    The input can be anything accepted by :meth:`OrderedDict`, 
    see :class:`dict`. All items must be the same length, which becomes the 
    length of the resulting recarray.
    
    >>> dict2rec(zip(("a", "b", "c"), 
    ...     (np.arange(0, 3), np.arange(10, 13), np.zeros(3))))
    array([(0, 10, 0.0), (1, 11, 0.0), (2, 12, 0.0)], 
          dtype=[('a', '<i...'), ('b', '<i...'), ('c', '<f8')])
    
    >>> dict2rec(OrderedDict(zip(("a", "b", "c"), 
    ...     (np.arange(0, 3), np.arange(10, 13), np.zeros(3)))))
    array([(0, 10, 0.0), (1, 11, 0.0), (2, 12, 0.0)], 
          dtype=[('a', '<i...'), ('b', '<i...'), ('c', '<f8')])
    
    >>> dict2rec({
    ...     "a": np.arange(0, 3), "b": np.arange(10, 13), "c": np.zeros((3,))})
    array([(0, 0.0, 10), (1, 0.0, 11), (2, 0.0, 12)],
          dtype=[('a', '<i...'), ('c', '<f8'), ('b', '<i...')])
    
    Vector-valued fields.
    
    >>> dict2rec([("a", [0, 1, 2]), 
    ...           ("b", [10, 11, 12]), 
    ...           ("c", [(0, 1), (2, 3), (4, 5)]),
    ...          ])
    array([(0, 10, [0, 1]), (1, 11, [2, 3]), (2, 12, [4, 5])], 
          dtype=[('a', '<i...'), ('b', '<i...'), ('c', '<i...', (2,))])
    """
    d = OrderedDict(*args, **kwargs)
    for k, v in d.items():
        d[k] = np.atleast_1d(v)
    # Convert keys to plain str because Numpy field names cannot be unicode
    dtype = [(str(k), v.dtype, v.shape[1:]) for k, v in d.items()]
    shape = len(v)
    x = np.zeros(shape=shape, dtype=dtype)
    for k, v in d.items():
        x[k] = v
    return x

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=
                    doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
