"""Find steady state or stable limit cycle of dynamic system."""

import numpy as np
from pysundials import cvode

class SteadyMixin(object):
    """Mixin to find steady state or stable limit cycle of dynamic system."""
    # Integrate until norm of rate-of-change < tol (steady state) or 
    # norm of difference between successive extrema < tol (stable limit cycle))

    def ydotnorm(self, tol):
        """
        Weighted root-mean-square norm of rate of change.
        
        See cvode.h and associated CVODE help pages.
        
        >>> class Test(SteadyMixin):
        ...     def __init__(self, ydot, reltol, abstol):
        ...         self.ydot = np.array(ydot)
        ...         self.reltol = np.array(reltol)
        ...         self.abstol = np.array(abstol)
        ...     def my_f_ode(self, t, y, ydot, g_data):
        ...         ydot[:] = self.ydot        
        """
        def result(t, y, gout, g_data=None):
            ydot = np.empty_like(y)
            self.my_f_ode(t, y, ydot, g_data)
            weights = 1.0 / (self.reltol * np.abs(y) + self.abstol)
            gout[0] = ((weights * ydot ** 2).sum() / weights.sum()) ** 0.5 - tol
            return 0
        return result
    
    def eq(self, tmax=None, tol=1e-4, last_only=True):
        """
        Find steady state.
        
        :param float tmax: Time limit for integration (default ``self.t[-1]``).
        :param float tol: Tolerance for weighted root-mean-square norm of 
            dy/dt. The WRMS norm is defined in cvode.h.
        :param float last_only: Include only the last time and state?
        
        .. plot::
            
            >>> from cgp import cvodeint
            >>> def ode(t, y, ydot, g_data):
            ...     ydot[:] = -y
            >>> class Test(cvodeint.Cvodeint, SteadyMixin):
            ...     pass            
            >>> test = Test(cvodeint.example_ode.logistic_growth, y=0.1)
            >>> t, y = test.eq(tmax=100, last_only=False)
            >>> plt.plot(t, y, '-', t[-1], y[-1], 'o')
        """
        g_rtfn = self.ydotnorm(tol)
        gout = np.zeros(1)
        g_rtfn(self.t, self.y, gout)
        if gout < 0:
            t = np.atleast_1d(self.tret)
            y = np.copy(self.y)
            flag = cvode.CV_ROOT_RETURN
        else:
            if tmax is None:
                tmax = self.t[-1]
            t, y, flag = self.integrate(t=tmax, nrtfn=1, g_rtfn=g_rtfn, 
                assert_flag=cvode.CV_ROOT_RETURN)
        if last_only:
            return t[-1], y[-1]
        else:
            return t, y, flag
