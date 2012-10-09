"""Cvode wrapper with named state variables and parameters."""

from contextlib import contextmanager
from collections import namedtuple

import numpy as np

from .core import Cvodeint
from ..utils.dotdict import Dotdict
from cgp.utils.rec2dict import rec2dict

class Namedcvodeint(Cvodeint):
    """
    Cvode wrapper with named state variables and parameters.
    
    Constructor arguments are as for :class:`~cgp.cvodeint.core.Cvodeint`, 
    except that *p* is a recarray. Further ``*args, **kwargs``
    are passed to :class:`~cgp.cvodeint.core.Cvodeint`.
        
    With no arguments, this returns the van der Pol model as an example.
        
    >>> Namedcvodeint()
    Namedcvodeint(f_ode=vanderpol, t=array([ 0.,  1.]), y=[-2.0, 0.0])
        
    .. todo:: 
        
        Should be::
    
            Namedcvodeint(f_ode=vanderpol, t=array([0, 1]), 
            y=rec.array([(-2.0, 0.0)], dtype=[('x', '<f8'), ('y', '<f8')]), 
            p=rec.array([(1.0,)], dtype=[('epsilon', '<f8')]))

    CVODE keeps state variables in a structure of class NVector, which does not 
    offer named fields. The ODE right-hand side must therefore refer to state 
    variables by index, or convert from/to a record array.
    
    .. todo:: The cleanest design would be to share memory between a record array 
       and the NVector, but I haven't got that to work.
    
    This example is similar to the one for :class:`~cgp.cvodeint.core.Cvodeint`. 
    Call the :class:`Namedcvodeint` constructor with arguments 
    ODE function, time, initial state and parameter array.
    
    >>> ode, t, y, p = Namedcvodeint.example()
    >>> n = Namedcvodeint(ode, t, y, p)
    
    The resulting object has a :class:`Recarraylink` to the state NVector.
    
    >>> n.yr.x
    array([-2.])
    
    Parameters can be read or written via the record array.
    
    >>> n.pr.epsilon
    array([ 1.])
    
    Note that the default initial state is a module-level variable shared by 
    all instances of this model object. Subsequent calls to 
    :meth:`Namedcvodeint` will see that the model is already imported and not 
    redo the initialization.
    
    Verify guard against inadvertently changing which object .pr or .y points to.
    
    >>> newpr = np.copy(n.pr)
    
    This is the way to do it.
    
    >>> n.pr[:] = newpr
    >>> n.integrate()
    (array([...], dtype=[('x', '<f8'), ('y', '<f8')]), 1)
    
    This isn't.
    
    >>> n.pr = newpr
    >>> n.integrate()
    Traceback (most recent call last):
    AssertionError:
            To change initial values or parameters for a model,
            use model.y[:] = ... or model.pr[:] = ...,
            not just model.y = ... or model.pr = ...
            The latter will cause the name model.y to point to a new object,
            breaking the link to the CVODE object.
            With Numpy arrays, using x[:] = ... is guaranteed to modify 
            contents only.
    """
    
    @staticmethod
    def example():
        """
        Example for :class:`Namedcvodeint`: van der Pol model.
        
        >>> Namedcvodeint.example()
        Example(ode=<function vanderpol at 0x...>, t=[0, 1], 
        y=rec.array([(-2.0, 0.0)], dtype=[('x', '<f8'), ('y', '<f8')]), 
        p=rec.array([(1.0,)], dtype=[('epsilon', '<f8')]))
        """
        t = [0, 1]
        y = np.rec.fromrecords([(-2.0, 0.0)], names="x y".split())
        p = np.rec.fromrecords([(1.0,)], names="epsilon")
        
        def vanderpol(t, y, ydot, f_data):
            """Van der Pol model."""
            ydot[0] = y[1]
            ydot[1] = p.epsilon * (1 - y[0] * y[0]) * y[1] - y[0]
        
        Nt = namedtuple("Example", "ode t y p")
        return Nt(vanderpol, t, y, p)
    
    def __init__(self, f_ode=None, t=None, y=None, p=None, 
        *args, **kwargs):
        if f_ode is None:
            f_ode, t_, y_, p_ = self.example()
            t = t_ if t is None else t
            y = y_ if y is None else y
            p = p_ if p is None else p
        if p is None:
            # Simplest array that allows copying and [:] assignment, etc.
            # Shape (), dtype float, no dtype.names
            p = np.zeros(0)
        super(Namedcvodeint, self).__init__(f_ode, t, y.view(float), 
            *args, **kwargs)
        self.yr = Recarraylink(self.y, y.dtype)
        self.pr = p
        # objects that should not be reassigned, but whose value may change
        self.reassignwarning = """
        To change initial values or parameters for a model, 
        use model.y[:] = ... or model.pr[:] = ..., 
        not just model.y = ... or model.pr = ...
        The latter will cause the name model.y to point to a new object, 
        breaking the link to the CVODE object.
        With Numpy arrays, using x[:] = ... is guaranteed to modify 
        contents only.
        """
        self.originals = dict(pr=self.pr, y=self.y, yr=self.yr)
        self.dtype = Dotdict(y=y.dtype, p=p.dtype)
    
    def ydoti(self, index):
        """
        Get rate-of-change of y[index] as a function of (t, y, gout, g_data).
        
        :param str_or_int index: Name or index of a state variable.
        
        For use with CVode's `rootfinding
        <https://computation.llnl.gov/casc/sundials/documentation/cv_guide/node3.html#SECTION00340000000000000000>`_
        functions.
        
        >>> vdp = Namedcvodeint()
        >>> gout = [None]
        >>> f = vdp.ydoti("y")
        >>> f(0, vdp.y, gout, None)  # returns 0 per CVODE convention
        0
        >>> gout  # The actual result is written to the output parameter gout
        [2.0]
        """
        # Get integer index if given as string
        try:
            index = self.dtype.y.names.index(index)
        except (NameError, ValueError):
            pass        
        return super(Namedcvodeint, self).ydoti(index)
    
    def integrate(self, **kwargs):
        """
        Return Cvodeint.integrate() of CellML model; convert state to recarray

        :parameters: See :meth:`cgp.cvodeint.core.Cvodeint.integrate`
        :return tuple:
            * **t** : time vector
            * **Yr** : state recarray. Yr[i] is state at time t[i]
              Yr.V is state variable V; Yr.V[i] is V at time t[i]
            * **flag** : last flag returned by CVode
        
        To convert the recarray to a normal array, use ``Y = Yr.view(float)``.

        >>> vdp = Namedcvodeint() # default van der pol model
        >>> t, Yr, flag = vdp.integrate(t=np.linspace(0, 20, 100))
        >>> Y = Yr.view(float)
        >>> Yr[0]
        rec.array([(-2.0, 0.0)], dtype=[('x', '<f8'), ('y', '<f8')])        
        >>> Y[0]
        array([-2.,  0.])
        >>> Yr.x
        array([[-2.        ], [-1.96634283], ... [-1.96940322], [-2.00814991]])
        """
        if not all(self.__dict__[k] is v for k, v in self.originals.items()):
            raise AssertionError(self.reassignwarning)
        t, Y, flag = super(Namedcvodeint, self).integrate(**kwargs)
        Yr = Y.view(self.dtype.y, np.recarray)
        return t, Yr, flag
    
    @contextmanager
    def autorestore(self, _p=None, _y=None, **kwargs):
        """
        Context manager to restore time, state and parameters after use.
        
        :param array_like _p: Temporary parameter vector
        :param array_like _y: Temporary initial state
        :param dict ``**kwargs``: dict of (name, value) for parameters or initial state
        
        In the following example, the assignment to epsilon and the call 
        to :meth:`~cgp.cvodeint.core.Cvodeint.integrate`
        change the parameters and state of the model object.
        This change is undone by the :meth:`autorestore` context manager.
        
        >>> vdp = Namedcvodeint()
        >>> before = np.array([vdp.yr.x, vdp.pr.epsilon])
        >>> with vdp.autorestore():
        ...     vdp.pr.epsilon = 2
        ...     t, y, flag = vdp.integrate()
        >>> after = np.array([vdp.yr.x, vdp.pr.epsilon])
        >>> all(before == after)
        True
        
        .. note:: Rootfinding settings cannot be restored, and so are cleared
           on exiting the .autorestore() context manager
        
        Optionally, you can specify initial parameters and state as _p and _y. 
        Further key=value pairs passed as arguments are used to update 
        parameters or state if the key exists in exactly one of them, 
        otherwise an error is raised.
        
        >>> pr = np.copy(vdp.pr).view(np.recarray)
        >>> y0 = np.copy(vdp.y).view(vdp.dtype.y, np.recarray)
        >>> pr.epsilon = 123
        >>> y0.x = 456
        >>> with vdp.autorestore(pr, y0, y=789):
        ...     vdp.pr.epsilon, vdp.yr.x, vdp.yr.y
        (array([ 123.]), array([ 456.]), array([ 789.]))
        >>> vdp.pr.epsilon, vdp.yr.x, vdp.yr.y
        (array([ 1.]), array([-2.]), array([ 0.]))
        
        The _p and _y variables can be record arrays or dict-like objects, 
        and do not need to specify all parameter fields.
        
        >>> with vdp.autorestore(_p=dict(epsilon=42)):
        ...     vdp.pr.epsilon
        array([ 42.])
        
        .. todo:: Refactor pr and y into properties/OrderedDict objects with 
           update() methods.
        
        Settings are restored even in case of exception.
        
        >>> with vdp.autorestore():
        ...     vdp.pr.epsilon = 50
        ...     raise Exception
        Traceback (most recent call last):
        Exception
        >>> bool(vdp.pr.epsilon == before[1])
        True
        """
        oldt, oldy, oldpar = np.copy(self.t), np.copy(self.y), np.copy(self.pr)
        if _p is not None:
            if np.asarray(_p).dtype.names:
                _p = rec2dict(_p)
            for k, v in _p.items():
                self.pr[k] = v
        if _y is not None:
            try:
                self.y[:] = _y
                # Assignment to NVector won't check number of elements, so:
                np.testing.assert_allclose(self.y, _y)
            except ValueError, exc:
                msg = "can only convert an array of size 1 to a Python scalar"
                assert msg in str(exc)
                self.y[:] = _y.squeeze()
            except TypeError: # float expected instead of numpy.void instance
                self.y[:] = _y.item()
        for k, v in kwargs.items():
            if k in self.dtype.p.names and k not in self.dtype.y.names:
                self.pr[k] = v
                continue
            if k in self.dtype.y.names and k not in self.dtype.p.names:
                # Recarraylink does not support item assignment
                setattr(self.yr, k, v)
                continue
            if k not in [self.dtype.y.names + self.dtype.p.names]:
                raise TypeError("Key %s not in parameter or rate vectors" % k)
            raise TypeError(
                "Key %s occurs in both parameter and state vectors" % k)
        self._ReInit_if_required(y=self.y)
        
        try:
            yield
        finally:
            self._ReInit_if_required(oldt, oldy)
            self.RootInit(0) # Disable any rootfinding
            self.pr[:] = oldpar
    
    @contextmanager
    def clamp(self, **kwargs):
        """
        Derived model with state value(s) clamped to constant values.
        
        Names and values of clamped variables are given as keyword arguments.
        Clamped state variables cannot be modified along the way.
        
        Use as a context manager.
        
        >>> vdp = Namedcvodeint()
        >>> with vdp.clamp(x=0.5) as clamped:
        ...     t, y, flag = clamped.integrate(t=[0, 10])
        
        The clamped model has its own instance of the CVODE integrator and 
        state NVector.
        
        >>> clamped.cvode_mem is vdp.cvode_mem
        False
        
        However, changes in state are copied to the original on exiting the 
        'with' block.
        
        >>> vdp.yr.x
        array([ 0.5])
        
        The clamped model has the same parameter array as the original.
        You will usually not change parameters inside the 'with' block, 
        but if you do, it will affect both models. Be careful.
        
        >>> with vdp.clamp(x=0.5) as clamped:
        ...     clamped.pr.epsilon = 2
        >>> vdp.pr.epsilon
        array([ 2.])
        
        Any hardcoded stimulus protocol is suppressed while clamped, 
        and restored afterwards.
        
        >>> t= [0, 1]
        >>> y = np.rec.fromrecords([(-2.0, 0.0)], names="x y".split())
        >>> p = np.rec.array([(1.0, 2.0)], names=["epsilon", "stim_amplitude"])
        >>> def test(t, y, ydot, f_data):
        ...     ydot[0] = y[1]
        ...     ydot[1] = p.epsilon * (1 - y[0] * y[0]) * y[1] - y[0]
        >>> model = Namedcvodeint(test, t, y, p)
        >>> model.pr.stim_amplitude
        array([ 2.])
        >>> with model.clamp(x=0) as clamped:
        ...     model.pr.stim_amplitude
        ...     clamped.pr.stim_amplitude
        array([ 0.])
        array([ 0.])
        >>> model.pr.stim_amplitude
        array([ 2.])
        
        The clamped model gets the same class as the original.
        
        >>> class Subclass(Namedcvodeint):
        ...    pass
        >>> model = Subclass()
        >>> with model.clamp(x=0) as clamped:
        ...     print model
        ...     print clamped
        Subclass(f_ode=vanderpol, t=array([ 0.,  1.]), y=[-2.0, 0.0])
        Subclass(f_ode=clamped, t=array([ 0.,  1.]), y=[0.0, 0.0])
        """
        # Indices to state variables whose rate-of-change will be set to zero
        i = np.array([self.dtype.y.names.index(k) for k in kwargs.keys()])
        v = np.array(kwargs.values())
        
        def clamped(t, y, ydot, f_data):
            """New RHS that prevents some elements from changing."""
            # CVODE forbids modifying the state directly...
            y_ = np.copy(y)
            # ...but may modify state variables even if ydot is always 0
            y_[i] = v
            self.f_ode(t, y_, ydot, f_data)
            ydot[i] = 0
            return 0
        
        # Initialize clamped state variables.
        y = np.array(self.y).view(self.dtype.y)
        for k, v in kwargs.items():
            y[k] = v
        
        # Use original options when rerunning the Cvodeint initialization.
        oldkwargs = dict((k, getattr(self, k)) 
            for k in "chunksize maxsteps reltol abstol".split())
        
        args, kwargs = self._init_args
        clamped_model = self.__class__(*args, **kwargs)
        Namedcvodeint.__init__(clamped_model, 
            clamped, self.t, y, self.pr, **oldkwargs)
        
        # Disable any hard-coded stimulus protocol
        if "stim_amplitude" in self.dtype.p.names:
            # self.pr.stim_amplitude is mutable, so cast to float
            old_stim_amplitude = float(self.pr.stim_amplitude)
            clamped_model.pr.stim_amplitude = 0
        
        try:
            yield clamped_model  # enter "with" block
        finally:
            # Copy values of state variables from clamped to original model
            for k in clamped_model.dtype.y.names:
                if k in self.dtype.y.names:
                    setattr(self.yr, k, getattr(clamped_model.yr, k))
            # Restore any hard-coded stimulus protocol
            if "stim_amplitude" in self.dtype.p.names:
                self.pr.stim_amplitude = old_stim_amplitude 
    
    def rates(self, t, y, par=None):
        """
        Compute rates for a given state trajectory.
        
        Unfortunately, the CVODE machinery does not offer a way to return rates 
        during integration. This function re-computes the rates at each time 
        step for the given state.
        
        >>> vdp = Namedcvodeint()
        >>> t, y, flag = vdp.integrate()
        >>> ydot = vdp.rates(t, y)
        >>> ydot[0], ydot[-1]
        (rec.array([(0.0, 2.0)], dtype=[('x', '<f8'), ('y', '<f8')]), 
         rec.array([(0.780218..., 0.513757...)], dtype=...))
        """
        t = np.atleast_1d(t).astype(float)
        y = np.atleast_2d(y).view(float)
        ydot = np.zeros_like(y)
        with self.autorestore(_p=par):
            for i in range(len(t)):
                self.f_ode(t[i], y[i], ydot[i], None)
        ydot = ydot.squeeze().view(self.dtype.y, np.recarray)
        return ydot

class Recarraylink(object):
    """
    Dynamic link between a Numpy recarray and any array-like object.
    
    Example: Use variable names to access elements of a Sundials state vector.
    We need to specify the Numpy dtype (data type), which in this case is 
    built by the :class:`~cellmlmodels.cellmlmodel.Cellmlmodel` constructor.
    
    >>> vdp = Namedcvodeint() # default example: van der Pol model
    >>> ral = Recarraylink(vdp.y, vdp.dtype.y)
    >>> ral
    Recarraylink([-2.0, 0.0], [('x', '<f8'), ('y', '<f8')])
    
    Now, a change made to either object is mirrored in the other:
    
    >>> ral.x = 123
    >>> vdp.y[1] = 321
    >>> vdp.y
    [123.0, 321.0]
    >>> ral
    Recarraylink([123.0, 321.0], [('x', '<f8'), ('y', '<f8')])
    
    Fixed bug: Previously, a direct modification of the array-like object was
    not applied to the recarray if the next Recarraylink operation was setting
    a named field. (The old value of the recarray was modified with the new
    field, and the direct modification was just forgotten. Afterwards, the
    wrong result overwrote the array-like object.)
    
    Here's an example relevant to Pysundials and CVODE.
    
    >>> y0 = np.array([0.0, 1.0, 2.0]) # default initial state
    >>> y = np.zeros(len(y0)) # state vector
    >>> ral = Recarraylink(y, dict(names=list("abc"), formats=[float]*len(y)))
    >>> y[:] = y0 # re-initialize state vector
    >>> ral.b = 42 # external modification
    >>> ral # reflects both changes
    Recarraylink([  0.  42.   2.], [('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
    
    (Before the bugfix, output was
    Recarraylink([  0.  42.   0.], [('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
    not reflecting the direct modification.)
    
    .. todo:: Recarraylink sometimes fails, perhaps for clamped models where I 
       change the NVector and cvode_mem.
    """
    def __init__(self, x, dtype):
        """Constructor. x : array-like object. dtype : numpy data type."""
        # use "object" class to avoid calling __getattr__ and __setattr__
        # before we're ready
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_xa", np.array(x))
        object.__setattr__(self, "_xr", self._xa.view(dtype).view(np.recarray))
    
    def __getattr__(self, attr):
        """Return recarray field after copying values from array-like object"""
        try:
            self._xa[:] = self._x # from array-like object to ndarray view 
            return self._xr[attr] # access attribute via recarray view
        except ValueError: # pass through attributes not in self._xr
            super(Recarraylink, self).__getattr__(attr)
    
    def __setattr__(self, attr, val):
        """Set recarray field and copy values to array-like object"""
        try:
            self._xa[:] = self._x # refresh ndarray in case of extern chg to _x
            self._xr[attr] = val # set attribute of recarray view
            self._x[:] = self._xa # copy ndarray view's values to array-like obj
        except ValueError: # pass through attributes not in self._xr
            super(Recarraylink, self).__setattr__(attr, val)
    
    def __array__(self):
        """
        Underlying record array for array operations.
        
        >>> np.copy(Recarraylink([-2.0, 0.0], [('x', '<f8'), ('y', '<f8')]))
        array([(-2.0, 0.0)], 
              dtype=[('x', '<f8'), ('y', '<f8')])
        
        Without __array__, output would have been
        array(Recarraylink([-2.0, 0.0], [('x', '<f8'), ('y', '<f8')]), 
            dtype=object)
        """
        return self._xr
    
    def __repr__(self):
        """
        Detailed string representation.
        
        >>> Recarraylink([123.0, 321.0], [('x', '<f8'), ('y', '<f8')])
        Recarraylink([123.0, 321.0], [('x', '<f8'), ('y', '<f8')])
        """
        return "Recarraylink(%s, %s)" % (self._x, self._xr.dtype)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=
        doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
