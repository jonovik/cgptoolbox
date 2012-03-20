"""Record array helper functions."""

import numpy as np

def cbind(*arrays):
    """
    Combine recarrays by columns.
    
    >>> a = np.array([(0, 1)], dtype=[("a", float), ("b", float)])
    >>> b = np.array([(2, 3)], dtype=[("c", float), ("d", float)])
    >>> cbind(a, b)
    array([(0.0, 1.0, 2.0, 3.0)], 
          dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8'), ('d', '<f8')])
    """
    if not arrays:
        raise TypeError("cbind() takes at least one argument (0 given)")
    dtype = np.dtype([d for a in arrays for d in a.dtype.descr])
    result = np.empty(shape=a.shape, dtype=dtype)  # pylint: disable=W0631
    for a in arrays:
        for k in a.dtype.names:
            # Squeeze to allow for trailing singleton dimensions
            result.squeeze()[k] = a.squeeze()[k]
    return result

from cgp.utils.unstruct import unstruct

def restruct(a, axes=0):
    """
    Convert shape dimensions to field dimensions.
    
    Converting a record array of shape (3,) with two scalar fields 
    to one of size () whose fields are shape (3,).
    
    >>> restruct(np.array([(0, 1), (2, 3), (4, 5)], 
    ...     dtype=[("a", "|i1"), ("b", "|i1")]))
    array(([0, 2, 4], [1, 3, 5]), dtype=[('a', '|i1', (3,)), ('b', '|i1', (3,))])
    
    This is an array of shape (2, 3) with two scalar fields.
    
    >>> a = np.array([[(0, 1), (2, 3),  (4,  5)],
    ...               [(6, 7), (8, 9), (10, 11)]], 
    ...               dtype=[('a', '|i1'), ('b', '|i1')])
    
    Folding the first dimension into each field gives an array of shape (3,) 
    whose fields have shape (2,).
    
    >>> restruct(a)
    array([([0, 6], [1, 7]), 
           ([2, 8], [3, 9]),
           ([4, 10], [5, 11])], dtype=[('a', '|i1', (2,)), ('b', '|i1', (2,))])
    """
    axes = np.atleast_1d(axes)
    u = unstruct(a)
    ax = range(u.ndim)
    # Move the given axes to the end
    for i in axes:
        ax.remove(i)
        ax.append(i)
    ut = u.transpose(ax).ravel()
    dtype = [(k, u.dtype, a.shape[axes]) for k in a.dtype.names]
    result = ut.view(dtype)
    result.shape = result.shape[:a.ndim - len(axes)]
    return result
