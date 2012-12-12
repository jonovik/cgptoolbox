"""Modify a function to return a default value in case of error."""

from functools import wraps
import logging
from contextlib import contextmanager

import numpy as np

class NullHandler(logging.Handler):
    """Null handler to use as default."""
    
    def emit(self, record):
        pass

logger = logging.getLogger("failwith")
logger.addHandler(NullHandler())

@contextmanager
def silenced(logger, level=logging.CRITICAL):  # pylint: disable=W0621
    """
    Silence a logger for the duration of the 'with' block.
    
    >>> import sys
    >>> logger = logging.Logger("test_silenced")
    >>> logger.addHandler(logging.StreamHandler(sys.stdout))
    >>> logger.error("Error as usual.")
    Error as usual.
    >>> with silenced(logger):
    ...     logger.error("Silenced error.")
    >>> logger.error("Back to normal.")
    Back to normal.
    
    You may specify a different temporary level if you like.
    
    >>> with silenced(logger, logging.INFO):
    ...     logger.error("Breaking through the silence.")
    Breaking through the silence.
    """
    oldlevel = logger.level
    try:
        logger.setLevel(level)
        yield logger
    finally:
        logger.setLevel(oldlevel)

def nans_like(x):
    """
    Returns an array of nans with the same shape and type as a given array.
    
    This also works recursively with tuples, lists or dicts whose leaf nodes 
    are arrays.
    
    >>> x = np.arange(3.0)
    >>> nans_like(x)
    array([ nan,  nan,  nan])
    >>> y = x.view([(k, float) for k in "a", "b", "c"])
    >>> nans_like(y)
    array([(nan, nan, nan)], dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
    >>> nans_like(y.view(np.recarray))
    rec.array([(nan, nan, nan)], dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
    
    Tuple, list, dict.
    
    >>> nans_like((x, y))
    [array([ nan,  nan,  nan]), array([(nan, nan, nan)], 
          dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])]
    >>> nans_like([x, y])
    [array([ nan,  nan,  nan]), array([(nan, nan, nan)], 
          dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])]
    >>> nans_like(dict(a=x, b=y))
    {'a': array([ nan,  nan,  nan]), 'b': array([(nan, nan, nan)], 
          dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])}
    
    Nested list and dict.
    
    >>> nans_like([x, [x, y]])
    [array([ nan, nan, nan]), [array([ nan, nan, nan]), array([(nan, nan, nan)],
                            dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])]]
    >>> nans_like(dict(a=x, b=dict(c=x, d=y)))
    {'a': array([ nan,  nan,  nan]), 
     'b': {'c': array([ nan,  nan,  nan]), 'd': array([(nan, nan, nan)], 
                dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])}}
    
    Note that there is no nan for integers.
    
    >>> nans_like((1, 2, 3))
    Traceback (most recent call last):
    AssertionError: nan is only defined for float types, not int...
    
    This works because the 1.0 makes Numpy interpret the tuple as a float array.
    
    >>> nans_like((1.0, 2, 3))
    array([ nan,  nan,  nan])
    """
    try:
        return dict((k, nans_like(v)) for k, v in x.iteritems())
    except AttributeError:
        try:
            xc = np.copy(x)
            try:
                xc = x.__array_wrap__(xc)
            except AttributeError:
                pass
            msg = "nan is only defined for float types, not %s" % xc.dtype
            assert not xc.dtype.kind == "i", msg
            xc.view(np.float).fill(np.nan)
            return xc
        except TypeError:
            return [nans_like(i) for i in x]

# pylint:disable=C0111
def failwith(default=None):
    """
    Modify a function to return a default value in case of error.
    
    >>> @failwith("Default")
    ... def f(x):
    ...     raise Exception("Failure")
    >>> f(1)
    'Default'
    
    Exceptions are logged, but the default handler doesn't do anything.
    This example adds a handler so exceptions are logged to :data:`sys.stdout`.
    
    >>> import sys
    >>> logger.addHandler(logging.StreamHandler(sys.stdout))
    >>> f(2)
    Failure in <function f at 0x...>. Default: Default. args = (2,), kwargs = {}
    Traceback (most recent call last):...
    Exception: Failure
    'Default'
    
    >>> del logger.handlers[-1] # Removing the handler added by the doctest
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception:  # pylint:disable=W0703
                msg = "Failure in %s. Default: %s. args = %s, kwargs = %s"
                logger.exception(msg, func, default, args, kwargs)
                result = default
            return result
        return wrapper
    return decorator

def failwithnanlikefirst(func):
    """
    Like :func:`failwith`, but the default is set to `nan` + result on first evaluation.
    
    >>> @failwithnanlikefirst
    ... def f(x):
    ...     return 1.0 / x
    >>> f(1)
    1.0
    >>> f(0)
    array(nan)
    
    Exceptions are logged, but the default handler doesn't do anything.
    This example adds a handler so exceptions are logged to :data:`sys.stdout`.
    
    >>> import sys
    >>> logger.addHandler(logging.StreamHandler(sys.stdout))
    >>> f(0)
    Failure in <function f at 0x...>. Default: nan. args = (0,), kwargs = {}
    Traceback (most recent call last):...
    ZeroDivisionError: float division...
    array(nan)
    
    If the first evaluation fails, the exception is logged with an explanatory 
    note, then re-raised. 
    
    >>> @failwithnanlikefirst
    ... def g():
    ...     raise Exception("Failure")
    >>> try:
    ...     g()                         
    ... except Exception, exc:
    ...     print "Caught exception:", exc 
    <function g at 0x...> failed on first evaluation, or result could not be 
    interpreted as array of float. args = (), kwargs = {}
    Traceback (most recent call last):...Exception: Failure
    Caught exception: Failure
    """
    d = {} # mutable container to store the default between evaluations
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not d:
            # First evaluation
            try:
                result = func(*args, **kwargs)
                d["default"] = nans_like(result)
            except Exception:  # pylint:disable=W0703
                msg = "%s failed on first evaluation, "
                msg += "or result could not be interpreted as array of float. "
                msg += "args = %s, kwargs = %s"
                logger.exception(msg, func, args, kwargs)
                raise
        else:
            # Not first evaluation, so default is defined
            try:
                result = func(*args, **kwargs)
            except Exception:  # pylint:disable=W0703
                msg = "Failure in %s. Default: %s. args = %s, kwargs = %s"
                logger.exception(msg, func, d["default"], args, kwargs)
                result = d["default"]
        return result
    return wrapper        

def failwithnan_asfor(*args, **kwargs):
    """
    Like :func:`failwith`, but the default is set to `nans_like(func(*args, **kwargs))`.
    
    >>> @failwithnan_asfor(2.0, 3)
    ... def f(value, length):
    ...     return [value] * length
    >>> f()
    array([ nan,  nan,  nan])
    """
    def decorator(func):
        default = nans_like(func(*args, **kwargs))
        return failwith(default)(func)
    return decorator

def failwithdefault_asfor(*args, **kwargs):
    """
    Like :func:`failwith`, but the default is set to `func(*args, **kwargs)`.
    
    >>> @failwithdefault_asfor(2, 3)
    ... def f(value, length):
    ...     return [value] * length
    >>> f()
    [2, 2, 2]
    """
    def decorator(func):
        default = func(*args, **kwargs)
        return failwith(default)(func)
    return decorator

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
