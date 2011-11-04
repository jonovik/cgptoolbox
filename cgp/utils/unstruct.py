"""
Unstructured view of a structured array whose fields are all the same type.
"""

import numpy as np

def unstruct(x):
    """
    Unstructured view of a structured array whose fields are all the same type.
    
    Usage:
    
    >>> fieldtype = np.int32   # ensure same result on 32- and 64-bit platforms
    >>> dtype = [("a", fieldtype), ("b", fieldtype)]
    >>> x = np.arange(2, dtype=fieldtype).view(dtype)
    >>> x
    array([(0, 1)], dtype=[('a', '<i4'), ('b', '<i4')])
    >>> unstruct(x) == np.array([[0, 1]])
    array([[ True,  True]], dtype=bool)
    
    Details:
    
    >>> x.shape
    (1,)
    >>> unstruct(x).shape
    (1, 2)
    >>> x = np.arange(4, dtype=fieldtype).view(dtype)
    
    Assignment to u will also affect x.
    
    >>> u = unstruct(x)
    >>> u[:] = 2
    >>> x
    array([(2, 2), (2, 2)], dtype=[('a', '<i4'), ('b', '<i4')])    
    
    (Some examples below use ELLIPSIS to allow for 64-bit Numpy tacking 
    ", dtype=np.int32" onto the output.)
    
    >>> x, unstruct(x), x.shape, unstruct(x).shape # doctest: +ELLIPSIS
    (array([(0, 1), (2, 3)], dtype=...), array([[0, 1], [2, 3]]...), (2,), (2, 2))
    >>> x = np.arange(24, dtype=fieldtype).view(dtype).reshape(4, 3)
    >>> x
    array([[(0, 1), (2, 3), (4, 5)],
           [(6, 7), (8, 9), (10, 11)],
           [(12, 13), (14, 15), (16, 17)],
           [(18, 19), (20, 21), (22, 23)]],
          dtype=[('a', '<i4'), ('b', '<i4')])
    >>> unstruct(x) # doctest: +ELLIPSIS
    array([[[ 0,  1], [ 2,  3], [ 4,  5]],
           [[ 6,  7], [ 8,  9], [10, 11]],
           [[12, 13], [14, 15], [16, 17]],    
           [[18, 19], [20, 21], [22, 23]]]...)
    >>> unstruct(x[0]) # doctest: +ELLIPSIS
    array([[0, 1], [2, 3], [4, 5]]...)
    
    Already unstructured types pass right through.
    
    >>> unstruct(np.arange(2))
    array([0, 1])
    
    Float types.
    
    >>> dtype = [("a", float), ("b", float)]
    >>> unstruct(np.arange(4.0).view(dtype))
    array([[ 0.,  1.], [ 2.,  3.]])
    
    Nonscalar fields are OK if all have the same shape.
    The result of unstruct(x) has shape (len(x), len(x.dtype), fieldshape).
    In this example, there are two records with three fields of shape four.
    
    >>> dtype = [("a", np.int32, 4), ("b", np.int32, 4), ("c", np.int32, 4)]
    >>> a = np.zeros(2, dtype=dtype)
    >>> a.view(np.int32)[:] = np.arange(a.nbytes / np.dtype(np.int32).itemsize)
    >>> a
    array([([0, 1, 2, 3],     [4, 5, 6, 7],     [8, 9, 10, 11]),
           ([12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23])],
      dtype=[('a', '<i4', (4,)),  ('b', '<i4', (4,)),  ('c', '<i4', (4,))])
    >>> unstruct(a)
    array([[[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
           [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    >>> unstruct(a).shape
    (2, 3, 4)
    
    Mixed types fail.
    
    >>> dtype = [("a", np.int8), ("b", np.int16)]
    >>> unstruct(np.zeros(3, dtype=dtype))
    Traceback (most recent call last):
    AssertionError: One or more fields has a different type or shape 
    than the first: [('a', '|i1'), ('b', '<i2')]
    
    Non-array objects are converted with np.array() if possible.
    
    >>> unstruct(range(5))
    array([0, 1, 2, 3, 4])
    
    List of arrays are OK if they can be concatenated.
    
    >>> unstruct([x[0,0], x[0,1]])                          # doctest: +ELLIPSIS
    array([[0, 1], [2, 3]])
    
    Zero-rank arrays are tricky; they can be structured, yet of type numpy.void.
    http://projects.scipy.org/numpy/wiki/ZeroRankArray
    Here, x0 and x1 are almost, but not completely, the same:
    
    >>> x0 = x[0, 0]
    >>> x1 = np.array((0, 1), dtype=[("a", fieldtype), ("b", fieldtype)])
    >>> (x0 == x1) and (x0.shape == x1.shape) and (x0.dtype == x1.dtype)
    True
    
    In spite of the equalities above, x0 and x1 are of different type.
    
    >>> x0
    (0, 1)
    >>> x1                                                  # doctest: +ELLIPSIS
    array((0, 1), dtype=[('a', '<i4'), ('b', '<i4')])
    >>> type(x0), type(x1)
    (<type 'numpy.void'>, <type 'numpy.ndarray'>)
    
    Unstructuring them was tricky, but finally works.
    
    >>> unstruct(x0)
    array([0, 1])
    >>> unstruct(x1)
    array([0, 1])
    """
    x = np.asanyarray(x)
    if x.dtype == object:
        x = np.concatenate([np.atleast_1d(i) for i in x])
    if x.dtype.fields is None:
        return x
    types = np.array([t[0] for t in x.dtype.fields.values()])
    fieldtype = types[0]  # dtype object
    msg = "One or more fields has a different type or shape than the first: %s"
    assert (types == fieldtype).all(), msg % x.dtype
    fieldshape = fieldtype.shape
    if fieldshape:
        fieldtype, _ = fieldtype.subdtype
    shape = x.shape + (len(types),) + fieldshape
    # Arrays of shape 0 cannot be .view()'ed, raising
    # ValueError: new type not compatible with array.
    # So promote to 1d once we're done computing the new shape.
    x = np.atleast_1d(x) # avoid ValueError
    xv = x.view(fieldtype)  # pylint: disable=E1103
    xv.shape = shape
    return xv

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
