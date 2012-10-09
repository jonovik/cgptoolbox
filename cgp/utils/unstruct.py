"""
Unstructured view of a structured array whose fields are all of the same type.
"""

import numpy as np

def unstruct(x):
    """
    Unstructured view of a structured array whose fields are all the same type.
    
    :param array_like x: Array, possibly with named fields.
    :return: Array with new dimension(s) appended for fields.
    
    Example: Unstructuring a 1-D array with two named fields.
    
    >>> x = np.array([(0, 1), (2, 3), (4, 5)], dtype=[('a', '<i4'), ('b', '<i4')])
    >>> u = unstruct(x)
    >>> u
    array([[0, 1], [2, 3], [4, 5]]...)
    
    The unstructured array has an extra dimension appended, with a length 
    equal to the number of fields in *x*.
    
    >>> x.shape
    (3,)
    >>> u.shape
    (3, 2)
    
    The result is a view of the original array, so modifications to *u* will 
    also affect *x*.
    
    >>> u.fill(2)
    >>> x
    array([(2, 2), (2, 2), (2, 2)], dtype=[('a', '<i4'), ('b', '<i4')])
    
    Already unstructured types pass right through.
    
    >>> unstruct(np.arange(2))
    array([0, 1])
    
    Nonscalar fields are OK if all have the same shape.
    The result of ``unstruct(x)`` has shape 
    ``(len(x), len(x.dtype), fieldshape)``.
    In this example, there are two records with three fields of shape four.
    
    >>> dtype = [("a", np.int32, (4,)), ("b", np.int32, (4,)), ("c", np.int32, (4,))]
    >>> a = np.zeros(2, dtype=dtype)
    >>> a.view(np.int32)[:] = np.arange(a.nbytes / np.dtype(np.int32).itemsize)
    >>> a
    array([([0, 1, 2, 3],     [4, 5, 6, 7],     [8, 9, 10, 11]),
           ([12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23])],
      dtype=[('a', '<i4', (4,)),  ('b', '<i4', (4,)),  ('c', '<i4', (4,))])
    >>> unstruct(a)
    array([[[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
           [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]...)
    >>> unstruct(a).shape
    (2, 3, 4)
    
    Mixed types fail.
    
    >>> dtype = [("a", np.int8), ("b", np.int16)]
    >>> unstruct(np.zeros(3, dtype=dtype))
    Traceback (most recent call last):
    AssertionError: One or more fields has a different type or shape 
    than the first: [('a', '|i1'), ('b', '<i2')]
    
    Non-array objects are converted with np.array() if possible. 
    List of arrays are OK if they can be concatenated.
    
    >>> unstruct(range(5))
    array([0, 1, 2, 3, 4])
    >>> unstruct([np.array([0, 1]), np.array([2, 3])])
    array([[0, 1], [2, 3]])
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
    x = np.atleast_1d(x)
    xv = x.view(fieldtype)
    return xv.reshape(shape)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
