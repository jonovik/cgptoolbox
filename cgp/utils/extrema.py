"""Find local maxima and minima of a 1-d array"""

import numpy as np
from numpy import sign, diff

__all__ = ["extrema"]

# pylint:disable=W0622
def extrema(x, max=True, min=True, withend=True):  #@ReservedAssignment
    """
    Return indexes, values, and sign of curvature of local extrema of 1-d array.
    
    The boolean arguments max, min, withend determine whether to 
    include maxima and minima, and include the endpoints.
    
    Basic usage.
    
    >>> x = [2, 1, 0, 1, 2]
    >>> extrema(x)   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    rec.array([(0, 2, -1), (2, 0, 1), (4, 2, -1)],
          dtype=[('index', '<i...'), ('value', '<i...'), ('curv', '<i...')])
    
    Options to include only certain types of extrema.
    
    >>> extrema(x, withend=False)
    rec.array([(2, 0, 1)],...
    >>> extrema(x, max=False)
    rec.array([(2, 0, 1)],...
    >>> extrema(x, min=False)
    rec.array([(0, 2, -1), (4, 2, -1)],...
    >>> extrema(x, max=False, min=False)
    rec.array([],...
    
    The beginning and end of flat segments both count as extrema, 
    except the first and last data point.
    
    >>> extrema([0, 0, 1, 1, 2, 2])
    rec.array([(1, 0, 1), (2, 1, -1), (3, 1, 1), (4, 2, -1)],...
    >>> extrema([0, 0, 0])
    rec.array([],...)
    >>> extrema([0, 0, 1, 1], withend=False)
    rec.array([(1, 0, 1), (2, 1, -1)],...
    
    @todo: Add options on how to handle flat segments.
    """
    x = np.squeeze(x) # ensure 1-d numpy array
    xpad = np.r_[x[1], x, x[-2]] # pad x so endpoints become minima or maxima
    curv = sign(diff(sign(diff(xpad)))) # +1 at minima, -1 at maxima
    i = curv.nonzero()[0] # nonzero() wraps the indices in a 1-tuple
    ext = np.rec.fromarrays([i, x[i], curv[i]], 
        names=["index", "value", "curv"])
    if not withend:
        ext = ext[(i > 0) & (i < len(x) - 1)]
    if not max:
        ext = ext[ext.curv >= 0]
    if not min:
        ext = ext[ext.curv <= 0]
    return ext

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
