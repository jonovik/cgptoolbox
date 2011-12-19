"""
Sample ODE functions for testing pysundials. Arguments are (t, y, ydot, f_data)

The computed result goes into ydot. The return value of the ODE function,
if any, is a flag to CVODE, see below.

The example ODEs are chosen to have simple, explicit solutions. 
For each ode(t, y, ydot, f_data), there is also an ode_sol(t, y0).

Sundials requires that ODE function returns 0 on success and <0 on 
unrecoverable error. However, you don't need to worry about this convention. 
Class :class:`~cgp.cvodeint.core.Cvodeint` will check that the ODE function 
follows the convention, and if not, wrap it in a 
:func:`~cvodeint.cvodefun.cvodefun` decorator to make it 
compatible. (:mod:`~cvodeint.cvodefun` is a separate, pure-Python module 
because Cython cannot yet compile nested functions.)

.. todo:: Cython 0.13+ handles closures, so this could be fixed.

Rootfinding functions are currently not wrapped like this.

>>> t = 0           # scalar time
>>> y = [2] # subscriptable state vector (usually numpy.ndarray; list works too)
>>> ydot = [0]      # writable output parameter
>>> f_data = None   # haven't figured out a way to use this yet...
>>> exp_growth(t, y, ydot, f_data)
>>> ydot
[2]

The functions accept optional parameters, e.g.:

>>> exp_growth(t, y, ydot, f_data, r=3)
>>> ydot
[6]

To pass parameters to one of these functions for use with pysundials:

>>> r = 4
>>> def my_exp_growth(t, y, ydot, f_data):
...     return exp_growth(t, y, ydot, f_data, r=r)
>>> my_exp_growth(t, y, ydot, f_data)
>>> ydot
[8]
"""

# Below, most of the functions return None, which is not printed, so that
# >>> fun(t, y, ydot, f_data); ydot
# first evaluates the function, displaying nothing for the None result, then
# displays ydot.

def g_rtfn_y(t, y, gout, g_data):
    """
    Trivial rootfinding function, sets ``gout[0] = y[0] - g_data``.
    
    >>> from pysundials import cvode
    >>> import ctypes
    >>> gout = cvode.NVector([0.0])
    >>> g_data = ctypes.c_float(2.5)
    >>> g_rtfn_y(0, [0], gout, ctypes.byref(g_data)), gout
    (0, [-2.5])
    >>> g_rtfn_y(3, [5], gout, ctypes.byref(g_data)), gout
    (0, [2.5])
    """
    import ctypes
    gout[0] = y[0] - ctypes.cast(g_data,
        ctypes.POINTER(ctypes.c_float)).contents.value
    return 0

def exp_growth(t, y, ydot, f_data, r=1):
    """
    Exponential growth equation.
    
    >>> t, y, ydot, f_data = 0, [2], [0], None
    >>> exp_growth(t, y, ydot, f_data); ydot
    [2]
    >>> exp_growth(t, y, ydot, f_data, r=3); ydot
    [6]
    """
    ydot[0] = r * y[0]

def exp_growth_sol(t, y0, r=1):
    """
    Solution to exponential growth equation.
    
    >>> from numpy import arange
    >>> exp_growth_sol(arange(4), 1).round(2)
    array([  1.  ,   2.72,   7.39,  20.09])
    """
    from numpy import exp
    return y0 * exp(r * t)

def nonsmooth_growth(t, y, ydot, f_data):
    """Exponential growth until t=1, then decay."""
    if t < 1:
        ydot[:] = y
    else:
        ydot[:] = - y

def nonsmooth_growth_sol(t, y0):
    """Solution to nonsmooth growth example."""
    import numpy as np
    t1 = t[t < 1]
    t2 = t[t >= 1]
    y1 = y0 * np.exp(t1)
    y2 = y0 * np.exp(1) / np.exp(t2 - 1)
    return np.r_[y1, y2]

def logistic_growth(t, y, ydot, f_data, r=1, K=1):
    """
    Logistic growth equation.
    
    >>> t, y, ydot, f_data = 0, [2], [0], None
    >>> logistic_growth(t, y, ydot, f_data); ydot
    [-2]
    >>> logistic_growth(t, y, ydot, f_data, r=3); ydot
    [-6]
    """
    ydot[0] = r * y[0] * (1 - y[0] / K)

def logistic_growth_sol(t, y0, r=1, K=1):
    """
    Solution to logistic growth equation.
    
    >>> from numpy import arange
    >>> logistic_growth_sol(arange(4), 0.1).round(2)
    array([ 0.1 ,  0.23,  0.45,  0.69])
    """
    from numpy import exp
    ert = exp(r * t)
    return  K * y0 * ert / (K + y0 * (ert - 1))

def const_growth(t, y, ydot, f_data, k=1):
    """
    Constant growth equation.
    
    >>> t, y, ydot, f_data = 5, [0], [0], None
    >>> const_growth(t, y, ydot, f_data); ydot
    [1]
    """
    ydot[0] = k

def const_growth_sol(t, y0, k=1):
    """
    Solution to constant growth equation.
    
    >>> from numpy import arange
    >>> const_growth_sol(arange(4), 0).round(2)
    array([0, 1, 2, 3])
    """
    return y0 + k * t

eps = [1]
def vdp(t, y, ydot, f_data):
    """van der Pol equation"""
    ydot[0] = y[1]
    ydot[1] = eps[0] * (1 - y[0] * y[0]) * y[1] - y[0]


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
