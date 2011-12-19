"""
n-dimensional gridding like Matlab's NDGRID

Typical usage:

>>> x, y, z = [0, 1], [2, 3, 4], [5, 6, 7, 8]
>>> X, Y, Z = ndgrid(x, y, z)

See ?ndgrid for details.
"""
import numpy as np

def ndgrid(*args, **kwargs):
    """
    n-dimensional gridding like Matlab's NDGRID
    
    The input *args are an arbitrary number of numerical sequences, 
    e.g. lists, arrays, or tuples.
    The i-th dimension of the i-th output argument 
    has copies of the i-th input argument.
    
    Optional keyword argument:
    same_dtype : If False (default), the result is an ndarray.
                 If True, the result is a lists of ndarrays, possibly with 
                 different dtype. This can save space if some *args 
                 have a smaller dtype than others.

    Typical usage:
    
    >>> x, y, z = [0, 1], [2, 3, 4], [5, 6, 7, 8]
    >>> X, Y, Z = ndgrid(x, y, z) # unpacking the returned ndarray into X, Y, Z

    Each of X, Y, Z has shape [len(v) for v in x, y, z].
    
    >>> X.shape == Y.shape == Z.shape == (2, 3, 4)
    True
    >>> X
    array([[[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
           [[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]])
    >>> Y
    array([[[2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4]],
           [[2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4]]])
    >>> Z
    array([[[5, 6, 7, 8],
            [5, 6, 7, 8],
            [5, 6, 7, 8]],
           [[5, 6, 7, 8],
            [5, 6, 7, 8],
            [5, 6, 7, 8]]])
    
    With an unpacked argument list:
    
    >>> V = [[0, 1], [2, 3, 4]]
    >>> ndgrid(*V) # an array of two arrays with shape (2, 3)
    array([[[0, 0, 0],
            [1, 1, 1]],
           [[2, 3, 4],
            [2, 3, 4]]])
    
    For input vectors of different data types, same_dtype=False makes ndgrid()
    return a list of arrays with the respective dtype.
    
    >>> ndgrid([0, 1], [1.0, 1.1, 1.2], same_dtype=False)
    [array([[0, 0, 0], [1, 1, 1]]), 
     array([[ 1. ,  1.1,  1.2], [ 1. ,  1.1,  1.2]])]
    
    Default is to return a single array.
    
    >>> ndgrid([0, 1], [1.0, 1.1, 1.2])
    array([[[ 0. ,  0. ,  0. ], [ 1. ,  1. ,  1. ]],
           [[ 1. ,  1.1,  1.2], [ 1. ,  1.1,  1.2]]])
    """
    same_dtype = kwargs.get("same_dtype", True)
    V = [np.array(v) for v in args] # ensure all input vectors are arrays
    shape = [len(v) for v in args] # common shape of the outputs
    result = []
    for i, v in enumerate(V):
        # reshape v so it can broadcast to the common shape
        # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        zero = np.zeros(shape, dtype=v.dtype)
        thisshape = np.ones_like(shape)
        thisshape[i] = shape[i]
        result.append(zero + v.reshape(thisshape))
    if same_dtype:
        return np.array(result) # converts to a common dtype
    else:
        return result # keeps separate dtype for each output

def expand_grid(*args):
    """
    A substitute for the R function expand.grid().
    
    Note that the first output vector varies fastest.
    
    >>> expand_grid([0, 1], [2, 3, 4])
    [array([0, 1, 0, 1, 0, 1]), array([2, 2, 3, 3, 4, 4])]
    """
    return [A.flatten() for A in ndgrid(*args[::-1])][::-1]

def gridrec(*kv, **kwargs):
    """
    Record array version of ndgrid. Try using it with .flat or .flatten().
    
    Usage example (fixing the dtype to ensure equal output on 32- and 64-bit).
    
    >>> xa = np.arange(3, dtype=np.int8)
    >>> xb = np.arange(2, dtype=np.int16)
    
    This can be called with any combination of (key, value) tuples or key=value 
    pairs, though the ordering of key=value pairs is arbitrary.
    
    >>> y = gridrec(("a", xa), b=xb)
    
    The result is a record array whose dtype has one field per input array. 
    and whose shape[i] equals the length of the ith input.
    
    >>> y
    rec.array([[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(2, 0), (2, 1)]],
    dtype=[('a', '|i1'), ('b', '<i2')])
    
    Individual fields have the same shape but a scalar dtype.
    
    >>> y.a, y.b
    (array([[0, 0], [1, 1], [2, 2]], dtype=int8), 
     array([[0, 1], [0, 1], [0, 1]], dtype=int16))
    
    You may use .flat or flatten() on the resulting record array if you just 
    want to loop over all items.
    
    >>> y.flatten()                                         # doctest: +ELLIPSIS
    rec.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], dtype=...)
    """
    k, v = zip(*(list(kv) + kwargs.items()))
    V = np.broadcast_arrays(*np.ix_(*v))
    return np.rec.fromarrays(V, names=k)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
