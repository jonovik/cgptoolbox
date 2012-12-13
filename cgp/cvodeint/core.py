#!/usr/bin/env python
"""
PySundials CVODE wrapper to automate routine steps of initializing, iterating...

Class :class:`Cvodeint` and its :meth:`~Cvodeint.integrate` method are the 
workhorses of this module. Importing with ``from cgp.cvodeint import *``
includes :class:`Cvodeint`, :exc:`CvodeException`, the :data:`flags` constants
and the :func:`cvodefun` decorator.

Simple example:

.. plot::
   :include-source:
   :width: 300
   
   from cgp.cvodeint import *
   cvodeint = Cvodeint(example_ode.exp_growth, t=[0,2], y=[0.1])
   t, y, flag = cvodeint.integrate()
   plt.plot(t, y, '.-')
   plt.xlabel("t")
   plt.ylabel("y")
   plt.title("$y = 0.1e^t$")

The module :mod:`.example_ode` defines some functions that are used in 
doctests. A replacement for :func:`scipy.integrate.odeint` is in module 
:mod:`.odeint`.

.. data:: flags

   CVODE return flags: dict of messages for each return value, as 
   returned by :func:`pysundials.cvode.CVodeGetReturnFlagName`.
"""

import traceback
import ctypes  # required for communicating with cvode
from pysundials import cvode
import numpy as np
import logging

__all__ = "CvodeException", "Cvodeint", "flags", "cvodefun"

# cdef inline double* bufarr(x):
#     """Fast access to internal data of ndarray"""
#     return <double*>(<np.ndarray>x).data

# logging
log = logging.getLogger("cvodeint")
log.addHandler(logging.StreamHandler())
# tab-delimited format string, see 
# http://docs.python.org/library/logging.html#formatter-objects
fmtstr = "%(" + ")s\t%(".join(
    "asctime levelname name lineno process message".split()) + ")s"
log.handlers[0].setFormatter(logging.Formatter(fmtstr))

# CVODE return flags: dict of messages for each return value. 
# See also cvode.CVodeGetReturnFlagName()
cv_ = dict([(k, v) for k, v in cvode.__dict__.iteritems() 
            if k.startswith("CV_")])
flags = dict([(v, [k1 for k1, v1 in cv_.iteritems() if v1==v]) 
    for v in np.unique(cv_.values())]) # example: flag = -12; print flags[flag]
del cv_

nv = cvode.NVector # CVODE vector data type


class CvodeException(StandardError):
    """
    :func:`pysundials.cvode.CVode` returned a flag not in 
    ``[CV_SUCCESS, CV_TSTOP_RETURN, CV_ROOT_RETURN]``
    
    The CvodeException object has a *result* attribute for 
    ``t, Y, flag`` = results so far.
    If the right-hand side function is decorated with 
    :func:`cvodefun` and raised an exception, the traceback is 
    available as ``ode.exc``, where ``ode`` is the wrapped function.
    """
    def __init__(self, flag_or_msg=None, result=None):
        if type(flag_or_msg) == int:
            flag = flag_or_msg
            message = "CVode returned %s" % (
                "None" if flag is None else cvode.CVodeGetReturnFlagName(flag))
        elif type(flag_or_msg) == str:
            message = flag_or_msg
        else:
            raise TypeError("Type int (flag) or str expected")
        super(CvodeException, self).__init__(message)
        self.result = result

def assert_assigns_all(fun, y, f_data=None):
    """
    Check that ``fun(t, y, ydot, f_data)`` does assign to all elements of *ydot*.
    
    Normal operation: no output, no exception.
    
    >>> def fun0(t, y, ydot, f_data):
    ...     ydot[0] = 0
    ...     ydot[1] = 0
    >>> assert_assigns_all(fun0, [0, 0]) # no output
    
    Raises exception on failure:
    
    >>> def fun1(t, y, ydot, f_data):
    ...     ydot[1] = 0 # fails to assign a value to ydot[0]
    >>> assert_assigns_all(fun1, [0, 0])
    Traceback (most recent call last):
    ...
    AssertionError: Function fun1 in module ... failed to assign finite value(s) to element(s) [0] of rate vector
    
    The exception instance holds the indices in attribute :attr:`i`.
    
    >>> try:
    ...     assert_assigns_all(fun1, [0, 0])
    ... except AssertionError, exc:
    ...     print exc.i
    [0]
    """
    y = np.array(y, ndmin=1)
    ydot = np.array(y, dtype=float, copy=True) # int(np.nan)==0, so need float
    ydot.fill(np.nan)
    fun(0, y, ydot, f_data)
    # Find elements that were either unassigned or not finite
    i = np.where(~np.isfinite(ydot))[0]
    if len(i) > 0:
        msg = "Function " + fun.__name__
        try:
            msg += " in module " + fun.__module__
        except AttributeError:
            pass
        msg += " failed to assign finite value(s)"
        msg += " to element(s) %s of rate vector" % i
        try:
            if fun.traceback:
                msg += ":" + fun.traceback
        except AttributeError:
            pass
        err = AssertionError(msg)
        err.i = i
        raise err

def cvodefun(fun):
    """
    Wrap ODE right-hand side function for use with CVODE in Pysundials.
    
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

def new_with_kwargs(cls, args, kwargs):
    """
    A helper function for pickling classes with keyword arguments.
    
    http://stackoverflow.com/questions/5238252/unpickling-new-style-with-kwargs-not-possible
    """
    return cls.__new__(cls, *args, **kwargs)

class Cvodeint(object):
    """
    Wrapper for common uses of :mod:`pysundials.cvode`
    
    :param function f_ode: Function of (t, y, ydot, f_data), 
        which writes rates-of-change to *ydot* and finishes with ``return 0``. 
        The ODE function must accept  four arguments, but is free to ignore 
        some of them.
    :param array_like t: Time, either [start end] or a vector of desired 
        return times. This determines whether 
        :meth:`~cgp.cvodeint.core.Cvodeint.integrate` returns time steps chosen 
        adaptively by CVODE, or just a copy of *t*.
    :param array_like y: Initial state vector, will be coerced to 
        cvode.NVector().
    :param reltol, abstol, nrtfn, g_rtfn, f_data, g_data: Arguments passed 
        to CVODE (`details 
        <https://computation.llnl.gov/casc/sundials/documentation/cv_guide/cv_guide.html>`_)
    :param int chunksize: When returning adaptive time steps, result 
        vectors are allocated in chunks of *chunksize*. 
    :param int maxsteps: If the number of  time-steps exceeds *maxsteps*, 
        an exception is raised.
    :param int mupper, mlower: Upper and lower bandwidth for the 
        `CVBand 
        <https://computation.llnl.gov/casc/sundials/documentation/cv_guide/node5.html#SECTION00566000000000000000>`_
        approximation to the Jacobian. CVDense is used by default if 
        *mupper* and *mlower* are both ``None``.
    
    **Usage example:**
    
    .. plot::
       :include-source:
       :width: 400
       
       >>> from cgp.cvodeint import *
       >>> from pysundials import cvode
       >>> from math import pi, sin
       
       Define the rate of change *dy/dt* as a function of the state *y*.
       The differential equation may use existing Python variables as 
       parameters. By convention, it must take arguments 
       *(t, y, ydot, f_data)*, where *ydot* is a writable parameter for 
       the actual result. See :func:`cvodefun` for details on CVODE's
       requirements for an ODE right-hand side.
       
       >>> r, K, amplitude, period = 1.0, 1.0, 0.25, 1.0
       >>> def ode(t, y, ydot, f_data):
       ...     ydot[0] = r * y[0] * (1 - y[0]/K) + amplitude * sin(2 * pi * t/period)
       
       Call the :meth:`Cvodeint` constructor with ODE function, time, and 
       initial state.
       
       >>> cvodeint = Cvodeint(ode, [0, 4], [0.1])

       Integrate over the time interval specified above.
       The :meth:`integrate` method returns the last *flag* returned by CVODE.
       
       >>> t, y, flag = cvodeint.integrate()
       
       >>> import matplotlib.pyplot as plt
       >>> h = plt.plot(t, y, '.-')
       >>> h = plt.title("Logistic growth + sinusoid")
       >>> text = "Last flag: " + cvode.CVodeGetReturnFlagName(flag)
       >>> h = plt.text(1, 0.8, text)
    
    .. note::
    
        *f_data* is intended for passing parameters to the ODE, but I couldn't
        get it to work. However, you can freely access variables that were in
        scope when the ODE function was defined, as this example shows.
    
    To change parameter values after the function is defined, use a mutable 
    variable for the parameter. In the first integration below, the state is 
    constant because the growth rate is zero.
    
    >>> growth_rate = [0.0]
    >>> def exp_growth(t, y, ydot, f_data):
    ...     ydot[0] = growth_rate[0] * y[0]
    ...     return 0
    >>> cvodeint = Cvodeint(exp_growth, t=[0, 0.5, 1], y=[0.25])
    >>> t, y, flag = cvodeint.integrate()
    >>> y
    array([[ 0.25], [ 0.25], [ 0.25]])
    
    Modify the growth rate and integrate further. Now, there is about 
    0.1 * 0.25 = 0.025 increase in y over 1 time unit.
    
    >>> growth_rate[0] = 0.1
    >>> t, y, flag = cvodeint.integrate(t=[0, 0.1, 1])
    >>> y.round(4)
    array([[ 0.25  ], [ 0.2525], [ 0.2763]])
    
    If the ode function is discontinuous, the solver may need to be
    reinitialized at the point of discontinuity.
    
    .. plot::
       :include-source:
       :width: 400
       
       from cgp.cvodeint import *
       eps = [1, 10]
       t_switch = 5
       t = [0, 10]
       def vdp(t, y, ydot, f_data): # van der Pol equation
           _eps = eps[0] if t <= t_switch else eps[1]
           ydot[0] = y[1]
           ydot[1] = _eps * (1 - y[0] * y[0]) * y[1] - y[0]
           return 0
       cvodeint = Cvodeint(vdp, t, [-2, 0])
       t0, y0, flag0 = cvodeint.integrate(t = [t[0], t_switch])
       t1, y1, flag1 = cvodeint.integrate(t = [t_switch, t[1]])
       plt.plot(t0, y0, t1, y1)    
    
    The :meth:`rootfinding <RootInit>` facilities of CVODE can be used 
    to find zero-crossings of a function of the state vector (example:
    :func:`~cgp.cvodeint.example_ode.g_rtfn_y`). Here we integrate the 
    van der Pol model to a series of specified values for y[0].
           
    .. plot::
       :include-source:
       :width: 400
       
       >>> from ctypes import byref, c_float
       >>> from cgp.cvodeint import *
       >>> thresholds = np.arange(-1.5, 1.5, 0.6)
       >>> cvodeint = Cvodeint(example_ode.vdp, [0, 20], [0, -2], reltol=1e-3)
       >>> res = [cvodeint.integrate(nrtfn=1, g_rtfn=example_ode.g_rtfn_y,
       ...     g_data=byref(c_float(thr))) for thr in thresholds]
       
       >>> import matplotlib.pyplot as plt
       >>> h = [plt.plot(t, y, '.-') for t, y, flag in res]
       >>> h = plt.hlines(thresholds, *plt.xlim(), color="grey")
    """
    """
    CVODE offers void pointers f_data and g_data for passing 
    parameters to ODE and rootfinding functions, respectively. Numpy objects 
    have ctypes properties that makes it easy to pass them to C functions, e.g.
    
    >>> cvode.CVodeSetFdata(cvodeint.cvode_mem, 
    ...     np.ctypeslib.as_ctypes(cvodeint.f_data))    # doctest: +SKIP 
    
    Here are some references on using the resulting pointer inside the ODE 
    function:
    `1 <http://article.gmane.org/gmane.comp.python.numeric.general/6827/>`__
    `2 <http://article.gmane.org/gmane.comp.python.scientific.devel/9331>`__
    `3 <http://article.gmane.org/gmane.comp.python.general/509212/>`__
    `4 <http://docs.python.org/library/struct.html>`__
    """  # pylint: disable=W0105
    def __init__(self, f_ode, t, y, reltol=1e-8, abstol=1e-8, nrtfn=None, 
        g_rtfn=None, f_data=None, g_data=None, chunksize=2000, maxsteps=1e4, 
        mupper=None, mlower=None):
        # Ensure that t and y can be indexed
        t = np.array(t, dtype=float, ndmin=1)
        try:
            y = np.array(y, dtype=float, ndmin=1)
        except ValueError:
            msg = "State vector y not interpretable as float: {}"
            raise ValueError(msg.format(y))
        assert len(y) > 0, "Empty state vector"
        # Ensure that f_ode assigns a value to all elements of the rate vector
        assert_assigns_all(f_ode, y, f_data)
        # Ensure that the function returns 0 on success and <0 on exception. 
        # (CVODE's convention is 
        # 0 = OK, >0 = recoverable error, <0 = unrecoverable error.)
        # If not, decorate as if with @cvodefun.
        self.f_ode = f_ode # store this for use in __repr__ etc.
        success_value = f_ode(t[0], nv(y), nv(y), f_data) # probably 0 or None
        try:
            error_value = f_ode(None, None, None, None) # <0 or raise exception
        except StandardError:
            error_value = None
            try:
                f_ode.traceback = ""
            except AttributeError:
                pass
        if (success_value == 0) and (error_value < 0):
            self.my_f_ode = f_ode
        else:
            self.my_f_ode = cvodefun(f_ode)
        # Variables y, tret, abstol are written by CVode functions, and their 
        # pointers must remain constant. They are assigned here; later 
        # assignments will copy values into the existing variables, like so:
        # self.tret.value = ...             (cvode.realtype)
        # self.y[:] = ...                   (cvode.NVector)
        self.y = nv(y)  # state vector (read/write)
        self.tret = cvode.realtype(0.0)  # actual time of return from solver
        if type(abstol) is cvode.realtype:
            self.abstol = abstol  # copy of abstol, used for ReInit
        elif np.isscalar(abstol):
            self.abstol = cvode.realtype(abstol)
        else:
            self.abstol = nv(abstol)
        self.t = t
        self.t0 = cvode.realtype(t[0]) # initial time
        self.tstop = t[-1] # end time
        self.n = len(y) # number of state variables
        self.reltol = reltol # copy of reltol, used for ReInit
        self.itol = cvode.CV_SV if (type(self.abstol) is nv) else cvode.CV_SS
        self.f_data = f_data # user data for right-hand-side of ODE
        self.g_data = g_data # user data for rootfinding function
        self.chunksize = chunksize
        self.maxsteps = maxsteps
        self.last_flag = None
        # CVODE solver object
        self.cvode_mem = cvode.CVodeCreate(cvode.CV_BDF, cvode.CV_NEWTON)
        cvode.CVodeMalloc(self.cvode_mem, self.my_f_ode, self.t0, self.y, 
            self.itol, self.reltol, self.abstol) # allocate & initialize memory
        if f_data is not None:
            cvode.CVodeSetFdata(self.cvode_mem, 
                np.ctypeslib.as_ctypes(self.f_data))
        cvode.CVodeSetStopTime(self.cvode_mem, self.tstop) # set stop time
        # Specify how the Jacobian should be approximated
        if mupper is None:
            cvode.CVDense(self.cvode_mem, self.n)
        else:
            cvode.CVBand(self.cvode_mem, self.n, mupper, mlower)        
        self.RootInit(nrtfn, g_rtfn, g_data)
    
    # pylint: disable=W0212
    def __new__(cls, *args, **kwargs):
        """Used for pickling."""
        instance = super(Cvodeint, cls).__new__(cls)
        instance._init_args = args, kwargs
        instance.__init__(*args, **kwargs)
        return instance
    
    def __reduce__(self):
        """Used for pickling."""
        args, kwargs = self._init_args
        return new_with_kwargs, (self.__class__, args, kwargs), None

    
    def ydoti(self, index):
        """
        Get rate-of-change of y[index] as a function of (t, y, gout, g_data).
        
        :param int index: Index of a state variable.
        
        For use with CVode's `rootfinding
        <https://computation.llnl.gov/casc/sundials/documentation/cv_guide/node3.html#SECTION00340000000000000000>`_
        functions.
        
        >>> from cgp.cvodeint.example_ode import vdp
        >>> c = Cvodeint(vdp, t=[0, 1], y=[1, 1])
        >>> gout = cvode.NVector([0.0])
        >>> f = c.ydoti(0)
        >>> f(0, c.y, gout, None)  # returns 0 per CVODE convention
        0
        >>> gout  # The actual result is written to the output parameter gout
        [1.0]
        """
        ydot = np.empty_like(self.y)
        
        def result(t, y, gout, g_data):  # pylint: disable=W0613
            """Function for CVODE rootfinding."""
            self.f_ode(t, y, ydot, None)
            gout[0] = ydot[index]
            return 0
        
        return result
        
    def integrate(self, t=None, y=None, nrtfn=None, g_rtfn=None, g_data=None, 
        assert_flag=None, ignore_flags=False):
        """
        Integrate over time interval, init'ing solver or rootfinding as needed.
        
        :param array_like t: new output time(s)
        :param array_like y: new initial state (default: resume from current 
            solver state
        :param int nrtfn: length of gout (required if *g_rtfn* is not ``None``)
        :param function g_rtfn: new rootfinding function of 
            *(t, y, gout, g_data)*,
            passed to :func:`~pysundials.cvode.CVodeRootInit`
        :param g_data: new data for rootfinding function
        :param int assert_flag: raise exception if the last flag returned by 
            CVode differs from *assert_flag* (see `flags`)
        :param bool ignore_flags: overrides assert_flag and does not check 
            CVode flag
        :return tuple: 
            * **tout**: time vector 
              (equal to input time *t* if that has len > 2), 
            * **Y**: array of state vectors at time ``t[i]``
            * **flag**: flag returned by :func:`~pysundials.cvode.CVode`, 
              one of ``[cvode.CV_SUCCESS, cvode.CV_ROOT_RETURN, 
              cvode.CV_TSTOP_RETURN]``
        
        * With no input arguments, integration continues from the current time 
          and state until the end time passed to :meth:`Cvodeint`.
        * If t is None, init solver and use the t passed to Cvodeint().
        * If t is scalar, integrate to that time, return adaptive time steps.
        * If len(t) == 1, return output only for that time.
        * If len(t) == 2, integrate from t[0] to t[1], return adaptive time 
          steps.
        * If len(t) > 2, integrate from t[0] and return fixed time steps = t.
        * If y is None, resume from current solver state.
        
        >>> from cgp.cvodeint.core import Cvodeint
        >>> from cgp.cvodeint.example_ode import exp_growth, exp_growth_sol
        >>> t, y = [0, 2], [0.1]
        >>> cvodeint = Cvodeint(exp_growth, t, y)
        
        With no parameters, :meth:`integrate` uses the current state, t0 and tstop:
        
        >>> t, Y, flag = cvodeint.integrate()
        
        Repeating the call will reset time:
        
        >>> t1, Y1, flag1 = cvodeint.integrate(); t1[0], t1[-1]
        (0.0, 2.0)
        
        Using a scalar or single-element t integrates from current time to t:
        
        >>> t2, Y2, flag2 = cvodeint.integrate(t=3); t2[0], t2[-1]
        (2.0, 3.0)
        
        Verify the solution:
        
        >>> Ysol = exp_growth_sol(t, Y[0])
        >>> err = (Y.squeeze() - Ysol)[1:]
        >>> "%.5e %.5e" % (err.min(), err.max())
        '2.75108e-09 1.25356e-06'
        
        ..  plot::
            :width: 300
            
            import matplotlib.pyplot as plt
            from cgp.cvodeint.core import Cvodeint
            from cgp.cvodeint.example_ode import exp_growth, exp_growth_sol
            cvodeint = Cvodeint(exp_growth, t=[0, 2], y=[0.1])
            t, Y, flag = cvodeint.integrate()
            Ysol = exp_growth_sol(t, Y[0])
            plt.plot(t, Y, '.-', t, Ysol, 'o')
            plt.title("Exponential growth")
            plt.xlabel("Time")
            plt.ylabel("y(t)")
        
        In case of error, the solution so far will be returned:
        
        ..  plot::
            :include-source:
            :width: 400
            
            from cgp.cvodeint import *
            import matplotlib.pyplot as plt
            import math
            def ode(t, y, ydot, f_data):
                ydot[0] = math.log(y[1])
                ydot[1] = 1 / y[0] - t # will eventually turn negative
            cvodeint = Cvodeint(ode, [0, 4], [1, 1])
            try:
                cvodeint.integrate()    # Logs error and prints traceback,   
            except CvodeException, exc: # both ignored by doctest.
                t, y, flag = exc.result
                plt.plot(t, y, '.-')
                plt.text(1, 1, "Flag: %s" % flags[flag])
        
        Another example:
        
        >>> from example_ode import logistic_growth, logistic_growth_sol
        >>> cvodeint = Cvodeint(logistic_growth, [0, 2], [0.1])
        >>> t, Y, flag = cvodeint.integrate()
        >>> Ysol = logistic_growth_sol(t, Y[0])
        >>> err = (Y.squeeze() - Ysol)[1:]
        >>> "%.5e %.5e" % (err.min(), err.max())
        '-1.40383e-07 3.40267e-08'        
        
        With len(t) == 2, re-initialize at t[0] and set tstop = t[-1]:
        
        >>> t, y, flag = cvodeint.integrate(t=[2, 3])
        >>> print np.array2string(t, precision=3)
        [ 2.     2.001  2.001  2.002  ... 2.966  2.995  3.   ]
        >>> print np.array2string(y, precision=3)
        [[ 0.451] [ 0.451] [ 0.451] ... [ 0.683] [ 0.69 ] [ 0.691]]
        
        With t omitted, re-initialize at existing t0 with current state:
        
        >>> t, y, flag = cvodeint.integrate(y=[0.2])
        >>> print np.array2string(t, precision=3)
        [ 2.     2.     2.001 ... 2.924  2.973  3.   ]
        >>> print np.array2string(y, precision=3)
        [[ 0.2 ] [ 0.2  ] [ 0.2  ] ... [ 0.387] [ 0.398] [ 0.405]]

        With both t and y given:
        
        >>> t, y, flag = cvodeint.integrate(t=[0, 1, 2], y=[0.3])
        >>> print np.array2string(y, precision=3)
        [[ 0.3  ] [ 0.538] [ 0.76 ]]
        
        Example with discontinuous right-hand-side:
        
        >>> eps = [1, 10]
        >>> t_switch = 5
        >>> tspan = [0, 10]
        >>> def vdp(t, y, ydot, f_data): # van der Pol equation
        ...     _eps = eps[0] if t <= t_switch else eps[1]
        ...     ydot[0] = y[1]
        ...     ydot[1] = _eps * (1 - y[0] * y[0]) * y[1] - y[0]
        ...     return 0
        >>> cvodeint = Cvodeint(vdp, t, [-2, 0])
        
        Integrate until RHS discontinuity:
        
        >>> t, y, flag = cvodeint.integrate(t = [tspan[0], t_switch])
        >>> t[-1], y[-1]
        (5.0, array([ 0.83707776, -1.30708838]))

        Calling :meth:`Cvodeint.integrate` again will do the necessary 
        re-initialization:
        
        >>> t, y, flag = cvodeint.integrate(t = [t_switch, tspan[1]])
        >>> t[0], t[-1], y[-1]
        (5.0, 10.0, array([-1.69...,  0.090...]))
        """
        self._ReInit_if_required(t, y)
        self.RootInit(nrtfn, g_rtfn, g_data)
        if len(self.t) > 2:
            result = self._integrate_fixed_steps()
        else:
            result = self._integrate_adaptive_steps()
        
        flag = result[-1]
        self.last_flag = flag
        # ensure assert_flag is iterable (or None):
        if type(assert_flag) is int:
            assert_flag = (assert_flag,)
        if ignore_flags or (assert_flag is None) or (flag in assert_flag):
            return result
        else:
            raise CvodeException(flag, result)
    
    def _ReInit_if_required(self, t=None, y=None):
        """
        Interpret/set time, state; call SetStopTime(), ReInit() if needed.
        
        If *t* is None, ``self.t`` is used to re-initialize at `
        ``tret==t0==self.t[0], tstop = self.t[-1]``.
        If *y* is ``None``, the current *y* is used at time ``tret``.
        If *t* is a scalar, *t0* is initialized to the current ``tret.value``.
        """
        # cdef long lptret = ctypes.addressof(self.tret)
        # cdef double* ptret = <double*>lptret
        # Cython: can use ptret[0] in place of self.tret
        if (t is None) and (self.last_flag ==
            cvode.CV_ROOT_RETURN) and (self.tret < self.tstop):
            t = self.tstop
        if t is not None:
            t = np.array(t, ndmin=1)
            if len(t) == 1:
                # @todo: Check if this is the internal step; we risk skipping
                # a beat after rootfinding. Simple check: same answer if
                # integrate(t = [last_time, next_time]) instead of 
                # integrate(next_time).
                self.t = [self.tret.value, t[0]]
                if y is None:
                    # CVodeGetDky returns nan if called before CVode().
                    # Restore the original self.y in this case.
                    Dky = nv(self.y)
                    cvode.CVodeGetDky(self.cvode_mem, self.tret, 0, self.y)
                    if any(np.isnan(self.y)):
                        self.y[:] = Dky
                else:
                    try:
                        self.y[:] = y
                    except TypeError:
                        # for numpy structured or record array
                        self.y[:] = y.item()
                # self.t = [cvode.CVodeGetCurrentTime(self.cvode_mem), t[0]]
            else:
                self.t = t
        if y is not None:
            try:
                self.y[:] = y
            except TypeError:
                self.y[:] = y.item() # for numpy structured or record array
        self.t0.value = self.t[0]
        self.tret.value = self.t[0] # needed for repeated integrate(t=None)
        self.tstop = self.t[-1]
        if (y is not None) or (t is None) or (len(self.t) >= 2):
            cvode.CVodeSetStopTime(self.cvode_mem, self.tstop)
            cvode.CVodeReInit(self.cvode_mem, self.my_f_ode, self.t0, self.y, 
                self.itol, self.reltol, self.abstol)
        # self.tret.value = cvode.CVodeGetCurrentTime(self.cvode_mem)

    def _integrate_adaptive_steps(self):
        """
        Repeatedly call CVode() with task CV_ONE_STEP_TSTOP and tout=tstop.
        
        Output: t, Y, flag. See Cvodeint.integrate().
        
        ..  plot::
            :include-source:
            :width: 400
            
            import matplotlib.pyplot as plt
            from cgp.cvodeint import *
            cvodeint = Cvodeint(example_ode.logistic_growth, 
                                t=[0, 2], y=[0.1], reltol=1e-3)
            t, y, flag = cvodeint.integrate()
            plt.plot(t, y, '.-')
        """
        Y = np.empty(shape=(self.chunksize, self.n)) # cdef np.ndarray
        t = np.empty(shape=(self.chunksize,)) # cdef np.ndarray
        d1 = self.chunksize # cdef int
        Y[0] = np.array(self.y, copy=True)
        t[0] = self.t0.value
        i = 1 # cdef int
        # cdef int flag
        # tret = self.tret
        maxsteps = self.maxsteps # cdef int
        # cdef long lptret = ctypes.addressof(tret)
        # cdef double* ptret = <double*>lptret
        # cdef double
        tstop = self.tstop
        cvode_mem = self.cvode_mem
        byref = ctypes.byref
        CVode = cvode.CVode
        y = self.y
        CV_SUCCESS = cvode.CV_SUCCESS # cdef int
        CV_ROOT_RETURN = cvode.CV_ROOT_RETURN # cdef int
        CV_TSTOP_RETURN = cvode.CV_TSTOP_RETURN # cdef int
        CV_ONE_STEP_TSTOP = cvode.CV_ONE_STE_TSTOP # typo in cvode # cdef int
        flag = None
        while self.tret < tstop:
            if i >= maxsteps:
                # truncate to drop unused array elements
                Y.resize((i, self.n), refcheck=False)
                t.resize(i, refcheck=False)
                raise CvodeException("Maximum number of steps exceeded", 
                                     (t, Y, flag))
            # solve ode for one internal time step
            # (pysundials has a typo in the name of the ONE_STEP_TSTOP constant)
            flag = CVode(cvode_mem, tstop, y, 
                byref(self.tret), CV_ONE_STEP_TSTOP)
            ## The top() function is hideously expensive, and gets evaluated 
            ## even if logging is set to ignore debug messages. Disable for now.
            # if (i % 10) == 0 and logging.DEBUG >= log.getEffectiveLevel():
            #     log.debug(top())
            if flag in (CV_SUCCESS, CV_TSTOP_RETURN, CV_ROOT_RETURN):
                # log.debug("OK: %s: %s" % (i, flags[flag]))
                Y[i], t[i] = y, self.tret.value # copy solver state & time
                if flag == CV_ROOT_RETURN:
                    i += 1
                    break
            else:
                log.debug("Exception: %s: %s" % (i, flags[flag]))
                # truncate to drop unused array elements
                Y.resize((i, self.n), refcheck=False)
                t.resize(i, refcheck=False)
                raise CvodeException(flag, (t, Y, flag))
            i += 1
            if i >= d1: # enlarge arrays with a new chunk
                d1 = len(t) + self.chunksize
                log.warning("Enlarging arrays from %s to %s" % (i, d1))
                Y.resize((d1, self.n), refcheck=False)
                t.resize(d1, refcheck=False)
        else: # if the while loop was skipped because self.tret >= tstop
            flag = CV_TSTOP_RETURN
        # truncate to drop unused array elements
        Y.resize((i, self.n), refcheck=False)
        t.resize(i, refcheck=False)
        return t, Y, flag

    def _integrate_fixed_steps(self):
        """
        Repeatedly call CVode() with task CV_ONE_STEP_TSTOP and tout=t[i]
        
        Output: t, Y, flag. See :meth:`integrate`.
        The *maxsteps* setting is ignored when using fixed time steps.
        
        >>> from example_ode import logistic_growth
        >>> cvodeint = Cvodeint(logistic_growth, t=[0, 0.5, 2], y=[0.1])
        >>> cvodeint.integrate()
        (array([ 0. , 0.5, 2. ]), array([[ 0.1 ], [ 0.154...], [ 0.45...]]), 0)
        
        ..  plot::
            :width: 400
            
            from example_ode import logistic_growth
            cvodeint = Cvodeint(logistic_growth, t=np.linspace(0, 2, 5), y=[0.1])
            t, y, flag = cvodeint.integrate()
            plt.plot(t, y, '.-')
        """
        imax = len(self.t)
        Y = np.empty(shape=(imax, self.n))
        t = np.empty(shape=(imax,))
        Y[0] = np.array(self.y).copy()
        t[0] = self.t0.value
        # tret = self.tret
        # cdef long lptret = ctypes.addressof(tret)
        # cdef double* ptret = <double*>lptret
        # cdef double
        cvode_mem = self.cvode_mem
        y = self.y
        # cdef double* pt = bufarr(self.t)
        i = 0
        for i in range(1, imax):
            # solve ode for one specified time step
            flag = cvode.CVode(cvode_mem, self.t[i], y, 
                ctypes.byref(self.tret), cvode.CV_NORMAL)
            Y[i], t[i] = y, self.tret.value # copy solver state & time
            if flag == cvode.CV_SUCCESS:
                continue
            else:
                break
        Y.resize((i + 1, self.n), refcheck=False)
        t.resize(i + 1, refcheck=False)
        result = t, Y, flag
        if flag in (cvode.CV_ROOT_RETURN, cvode.CV_SUCCESS):
            return result
        else:
            raise CvodeException(flag, result)
    
    def RootInit(self, nrtfn, g_rtfn=None, g_data=None):
        """
        Initialize rootfinding, disable rootfinding, or keep current settings.
        
        * If nrtfn == 0, disable rootfinding.
        * If nrtfn > 0, set rootfinding for g_rtfn.
        * If nrtfn is None, do nothing.
        
        Details `here 
        <https://computation.llnl.gov/casc/sundials/documentation/cv_guide/node6.html#SECTION00670000000000000000>`__.
        
        >>> from pysundials import cvode
        >>> from example_ode import exp_growth, g_rtfn_y
        >>> g_data = ctypes.c_float(2.5)
        >>> cvodeint = Cvodeint(exp_growth, t=[0, 3], y=[1],
        ...     nrtfn=1, g_rtfn=g_rtfn_y, g_data=ctypes.byref(g_data))
        >>> t, y, flag = cvodeint.integrate()
        >>> y[-1]
        array([ 2.5])
        >>> cvode.CVodeGetReturnFlagName(flag)
        'CV_ROOT_RETURN'

        Warn if nrtfn is not given but g_rtfn or g_data is given:
        
        >>> cvodeint = Cvodeint(exp_growth, t=[0, 3], y=[1],
        ...     g_rtfn=g_rtfn_y, g_data=ctypes.byref(g_data))
        Traceback (most recent call last):
        ...
        CvodeException: If g_rtfn or g_data is given, nrtfn is required.
        """
        if nrtfn is not None:
            cvode.CVodeRootInit(self.cvode_mem, int(nrtfn), g_rtfn, g_data)
        elif (g_rtfn is not None) or (g_data is not None):
            raise CvodeException(
                "If g_rtfn or g_data is given, nrtfn is required.")
    
    def __repr__(self):
        """
        String representation of Cvodeint object
        
        Unfortunately not as detailed under Cython as in pure Python.
        """
        import inspect
        try:
            # Ignore *args, **kwargs
            args, _, _, defaults = inspect.getargspec(self.__init__)
        except TypeError: # "arg is not a Python function" problem in Cython
            msg = "%s %s\ngetargspec(__init__) is not available"
            msg += " when running under Cython"
            return msg % (self.__class__, self.f_ode)
        if defaults is None:
            defaults = ()
        del args[0] # remove self from argument list
        # pad list of defaults to same length as args
        defaults = [None] * (len(args) - len(defaults)) + list(defaults)
        arglist = []
        for arg, default in zip(args, defaults):
            try:
                val = getattr(self, arg)
                if inspect.isfunction(val):
                    arglist.append((arg, val.func_name))
                elif val != default:
                    arglist.append((arg, repr(val)))
            except ValueError:
                try:                
                    if any(val != default):
                        arglist.append((arg, repr(val)))
                except ValueError:
                    if np.any(val != default):
                        arglist.append((arg, repr(val)))
            except AttributeError:
                continue
        argstr = ", ".join(["%s=%s" % x for x in arglist])
        return "%s(%s)" % (self.__class__.__name__, argstr)

    def __eq__(self, other):
        """
        Compare a CVODE wrapper object to another.
        
        >>> from example_ode import vdp, exp_growth 
        >>> a, b = [Cvodeint(vdp, [0, 20], [0, -2]) for i in range(2)]
        >>> a == b
        True
        >>> a.y[0] = 1
        >>> a == b
        False
        >>> c = Cvodeint(exp_growth, t=[0,2], y=[0.1])
        >>> a == c
        False
        """        
        return repr(self) == repr(other)
    
    def diff(self, other):
        """
        Return a detailed comparison of a CVODE wrapper object to another.
        
        Returns 0 if the objects are equal, otherwise a string.
        
        >>> from example_ode import vdp, exp_growth
        >>> from pprint import pprint 
        >>> a, b = [Cvodeint(vdp, [0, 20], [0, -2]) for i in range(2)]
        >>> print a.diff(b)
        <BLANKLINE>
        >>> a.y[0] = 1
        >>> print a.diff(b)
        - Cvodeint(f_ode=vdp, t=array([ 0., 20.]), y=[1.0, -2.0],
        ?                                           ^
        + Cvodeint(f_ode=vdp, t=array([ 0., 20.]), y=[0.0, -2.0],
        ?                                           ^
        >>> c = Cvodeint(exp_growth, t=[0,2], y=[0.25])
        >>> print a.diff(c)
        - Cvodeint(f_ode=vdp, t=array([  0.,  20.]), y=[1.0, -2.0],
        ?                ^^             -      -        ^ ---- ^^
        + Cvodeint(f_ode=exp_growth, t=array([ 0.,  2.]), y=[0.25],
        ?                ^^ +++++++                          ^  ^
        """
        import textwrap, difflib
        s, o = [textwrap.wrap(repr(x)) for x in self, other]
        return "\n".join([li.strip() for li in difflib.ndiff(s, o) 
                          if not li.startswith(" ")])

if __name__ == "__main__":
    import doctest
    failure_count, test_count = doctest.testmod(
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    print """
        NOTE: This doctest will (by design) result in an error message and a 
        traceback, which will be ignored by doctest (apparently, is is printed 
        outside of standard error and output). 
        Warnings about "enlarging arrays" are also intended.
        """
    if failure_count == 0:
        print """All doctests passed."""
