"""Wrap ODE right-hand side function for use with CVODE in Pysundials"""
__all__ = ["cvodefun"]

import sys
import traceback

def cvodefun(fun):
    """
    Make any callable into a CVRhsFn, returning -1 on exception and 0 otherwise.
    
    A `CVRhsFn 
    <https://computation.llnl.gov/casc/sundials/documentation/cv_guide/node5.html#SECTION00561000000000000000>`_ 
    (CVODE right-hand side function) is called for its side effect,
    modifying the *ydot* output vector. It should return 0 on success,
    a positive value if a recoverable error occurs,
    and a negative value if it failed unrecoverably.
    
    This decorator allows you to code the CVRhsFn without explicitly assigning
    the return value. It returns a callable object that returns -1 on exception
    and 0 otherwise. In case of exception, the traceback text is stored as a
    "traceback" attribute of the object.
    
    Below, :func:`ode` does not return anything. 
    It raises an exception if y[0] == 0.
    
    >>> @cvodefun
    ... def ode(t, y, ydot, f_data):
    ...     ydot[0] = 1 / y[0]
    >>> ydot = [None]
    >>> ode(0, [1], ydot, None)
    0
    >>> ydot
    [1]
    >>> ode(0, [0], ydot, None)
    -1
    >>> print ode.traceback
    Traceback (most recent call last):
    ...
    ZeroDivisionError: integer division or modulo by zero

    The wrapped function is a proper CVRhsFn with a return value. 
    In case of exception, the return value is set to -1, otherwise the return 
    value is passed through.
    
    >>> @cvodefun
    ... def ode(t, y, ydot, f_data):
    ...     ydot[0] = 1 / y[0]
    ...     return y[0]
    
    >>> ode
    @cvodefun wrapper around <function ode at 0x...>
    >>> ode(0, [2], ydot, None)
    2
    >>> ode(0, [-2], ydot, None)
    -2
    >>> ode(0, [0], ydot, None)
    -1
    >>> print ode.traceback
    Traceback (most recent call last):
    ...
    ZeroDivisionError: integer division or modulo by zero
    
    Pysundials relies on the ODE right-hand side behaving like a function.
    
    >>> ode.__name__
    'ode'
    >>> ode.func_name
    'ode'
    """
    class odefun(object):
        """Wrapper for a CVODE right-hand side function"""
        def __init__(self):
            self.__name__ = fun.__name__ # used by pysundials/cvode.py
            self.func_name = fun.__name__ # used by pysundials/cvode.py
            self.traceback = ""
        def __call__(self, *args, **kwargs):
            """Return function value if defined, -1 if exception, 0 otherwise"""
            self.traceback = ""
            try:
                result = fun(*args, **kwargs)
                if result is None:
                    return 0
                else:
                    return result
            except StandardError: # allow KeyboardInterrupt, etc., to work
                self.traceback = traceback.format_exc()
                return -1
        def __repr__(self):
            return "@cvodefun wrapper around %s" % fun
    return odefun()

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
