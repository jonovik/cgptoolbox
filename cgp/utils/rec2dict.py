"""
Convert recarray to ordered dictionary.

This module was motivated by the need to export recarrays to R via rnumpy.
Using an OrderedDict preserves the order of lists.
"""
from operator import itemgetter
import numpy as np
from ordereddict import OrderedDict

def rec2dict(x):
    """
    Convert recarray (or structured ndarray) to dictionary.
    
    See below for passing (nested) record arrays to R.
    
    Basic usage:
    
    >>> import numpy as np
    >>> A = np.rec.fromrecords([(10, 1.5), (20, 2.5)], names='id, x')
    >>> A                             # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    rec.array([(10, 1.5), (20, 2.5)], dtype=[('id', '<i...'), ('x', '<f8')])
    >>> rec2dict(A)
    OrderedDict([('id', array([10, 20])), ('x', array([ 1.5,  2.5]))])
    
    This works even with nested record arrays:
    
    >>> nested_dtype = [('id', '<i8'), ('x', '<f8'),
    ...     ('rec', [('a', '<i8'), ('b', '<f8')])]
    >>> B = np.rec.fromrecords([(10, 1.5, (1, 2)), (20, 2.5, (3, 4))],
    ...     dtype=nested_dtype)
    
    (Using +ELLIPSIS because 10 prints as 10L on some versions.) 
    
    >>> B                             # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    rec.array([(10..., 1.5, (1..., 2.0)), (20..., 2.5, (3..., 4.0))],
          dtype=[('id', '<i8'), ('x', '<f8'), 
                 ('rec', [('a', '<i8'), ('b', '<f8')])])
    
    (Using +ELLIPSIS because some versions include ", dtype=int64" for the 
    integer arrays.)
    
    >>> rec2dict(B)                   # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    OrderedDict([('id', array([10, 20]...)), 
                 ('x', array([ 1.5,  2.5])), 
                 ('rec', OrderedDict([('a', array([1, 3]...)), 
                                      ('b', array([ 2.,  4.]))]))])
    
    Passing a record array (possibly nested) to R:
    
    >>> from rnumpy import *                                    # doctest: +SKIP
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

def dict2rec(d):
    """
    Convert dict of arrays to nested recarray.
    
    The input can be anything accepted by OrderedDict(), see ?dict.
    
    >>> dict2rec(zip(("a", "b", "c"), 
    ...     (np.arange(0, 3), np.arange(10, 13), np.zeros(3))))
    ... # doctest: +ELLIPSIS
    array([(0, 10, 0.0), (1, 11, 0.0), (2, 12, 0.0)], 
          dtype=[('a', '<i...'), ('b', '<i...'), ('c', '<f8')])
    
    >>> dict2rec(OrderedDict(zip(("a", "b", "c"), 
    ...     (np.arange(0, 3), np.arange(10, 13), np.zeros(3)))))
    ... # doctest: +ELLIPSIS
    array([(0, 10, 0.0), (1, 11, 0.0), (2, 12, 0.0)], 
          dtype=[('a', '<i...'), ('b', '<i...'), ('c', '<f8')])
    
    >>> dict2rec({
    ...     "a": np.arange(0, 3), "b": np.arange(10, 13), "c": np.zeros((3,))})
    ... # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([(0, 0.0, 10), (1, 0.0, 11), (2, 0.0, 12)],
          dtype=[('a', '<i...'), ('c', '<f8'), ('b', '<i...')])
        
    @todo: Make this work if numeric fields have more than one column.
    
    >>> d = {"a": np.arange(0, 3), "b": np.arange(10, 13), "c": np.zeros((3, 2))}
    >>> d # DOES NOT WORK YET
    {'names': ('a', 'b', 'c'), 'formats': [dtype('int64'), dtype('int64'), dtype('float64')]}
    >>> dict2rec(d) # DOES NOT WORK YET
    """
    d = OrderedDict(d)
    for k, v in d.items():
        d[k] = np.atleast_1d(v)
    dtype = [(k, v.dtype) for k, v in d.items()]
    shape = len(v)
    x = np.zeros(shape=shape, dtype=dtype)
    for k, v in d.items():
        x[k] = v
    return x

if __name__ == "__main__":
    import doctest
    doctest.testmod()
