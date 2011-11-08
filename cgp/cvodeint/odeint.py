"""
Simple replacement for :func:`scipy.integrate.odeint` based on Sundials.

See :func:`odeint`.
"""

from .core import Cvodeint

def odeint(func, y0_in, t_in, args=()):
    """
    Simple replacement for scipy.integrate.odeint() based on Sundials.
    
    Example from http://www.scipy.org/SciPyPackages/Integrate
    
    >>> from scipy import *
    >>> from pylab import *
    >>> deriv = lambda y,t : array([y[1],-y[0]])
    >>> # Integration parameters
    >>> start=0
    >>> end=10
    >>> numsteps=10000
    >>> time=linspace(start,end,numsteps)
    >>> from scipy import integrate
    >>> y0=array([0.0005,0.2])
    >>> y=integrate.odeint(deriv,y0,time)
    >>> plot(time,y[:,0])                                  # doctest: +SKIP
    >>> show()                                             # doctest: +SKIP
    >>> y2=odeint(deriv,y0,time)                           # should equal y
    >>> print "%.5e" % abs((y - y2).max())                 # should be small
    6.82502e-07
    """
    def ode(t, y, ydot, f_data):
        """Right-hand side."""
        ydot[:] = func(y, t, args) if args else func(y, t)
        return 0
    cvodeint = Cvodeint(ode, t_in, y0_in)
    result = cvodeint.integrate()
    return result[1]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
