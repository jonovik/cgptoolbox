"""Functions for trimming invalid or extreme values from a vector."""

import numpy as np

def trimlim(x, low=0, high=0):
    """
    Return lower and upper bound for trimming a vector.
    
    >>> trimlim(np.linspace(100, 200, 101), 0.25, 0.5)
    (125.0, 149.0)
    >>> trimlim(np.linspace(100, 200, 101), 25, 0.5)
    (125.0, 149.0)
    """
    x = np.sort(x)
    x = x[np.isfinite(x)]
    n = len(x)
    # Convert low and high to integer in [0, n]
    low = round(n * low) if (0 < low < 1) else low
    high = round(n * high) if (0 < high < 1) else high
    low = min(n - 1, low)
    high = n - 1 - high
    if high < 0:
        high = 0
    return x[low], x[high]

def trim(x, low=0, high=0):
    """
    Return x[np.isfinite(x)] with extremes removed.
    
    x : vector
    low : number (if integer) or proportion (if float) of low values to trim
    high : number (if integer) or proportion (if float) of high values to trim
    
    Trimming the last half and the first two elements.
    
    >>> trim(range(10), low=2, high=0.5)
    array([2, 3, 4])
    
    Trimming the first quarter and last half of the interval from 100 to 200.
    
    >>> trim(np.linspace(100, 200, 101), 0.25, 0.50) # doctest: +ELLIPSIS
    array([ 125.,  126.,  ...,  148.,  149.])
    
    If there are duplicate values, the number or proportion cut off may be less 
    than requested. For example, even if "high" is large, the elements equal to 
    min(x) are all returned.
    
    >>> trim([1] * 3, high=10)
    array([1, 1, 1])
    
    If either low or high is 0, at least one element will be returned.
    
    >>> trim(range(10), high=9), trim(range(10), high=10)
    (array([0]), array([0]))
    >>> trim(range(10), low=9), trim(range(10), low=10)
    (array([9]), array([9]))
    
    Setting both low and high may return an empty array.
    
    >>> trim(range(10), low=5, high=5) # doctest: +ELLIPSIS
    array([], dtype=...)
    """
    x = np.array(x)
    xlow, xhigh = trimlim(x, low, high)
    return x[np.isfinite(x) & (xlow <= x) & (x <= xhigh)]

def trimpair(x, low=0, high=0):
    """
    Return rows of x where all columns are finite, possibly removing extremes.
    
    >>> np.random.seed(42)
    >>> x = (100 * np.random.random((5,2))).round()
    >>> x[2,0], x[3, 1] = np.nan, np.nan
    >>> trimpair(x)
    array([[ 37.,  95.],
           [ 73.,  60.],
           [ 60.,  71.]])
    
    """
    x = np.array(x)
    x = x[np.isfinite(x).all(axis=1), :]
    xlow, xhigh = np.transpose([trimlim(col, low, high) for col in x.T])
    return x[(xlow <= x).all(axis=1) & (x <= xhigh).all(axis=1), :]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
