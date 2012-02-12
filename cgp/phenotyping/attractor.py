"""Find steady state or stable limit cycle of dynamic system."""

from collections import deque

import numpy as np
from pysundials import cvode
from cgp.virtexp.elphys.clampable import catrec

class SteadyMixin(object):
    """Mixin to find steady state or stable limit cycle of dynamic system."""
    # Integrate until norm of rate-of-change < tol (steady state) or 
    # norm of difference between successive extrema < tol (stable limit cycle))
    
    def weighted_rms(self, y):
        """Norm of vector using CVODE weights based on reltol and abstol."""
        weights = 1.0 / (self.reltol * np.abs(y.view(float)) + self.abstol)
        return ((weights * y ** 2).sum() / weights.sum()) ** 0.5
    
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
        ydot = np.empty_like(self.y)
        
        def result(t, y, gout, g_data=None):
            self.my_f_ode(t, y, ydot, g_data)
            gout[0] = self.weighted_rms(ydot) - tol
            return 0
        
        return result
    
    def eq(self, tmax=None, tol=1e-4, last_only=True):
        """
        Find steady state.
        
        :param float tmax: Time limit for integration (default ``self.t[-1]``).
        :param float tol: Tolerance for weighted root-mean-square norm of 
            dy/dt. The WRMS norm is defined in cvode.h.
        :param float last_only: Include only the last time and state?
        
        >>> from cgp import cvodeint
        >>> class Test(cvodeint.Cvodeint, SteadyMixin):
        ...     pass            
        >>> test = Test(cvodeint.example_ode.logistic_growth, t=[0, 20], y=0.1)
        >>> test.eq()
        (11.407..., array([ 0.999...]))
        
        .. plot::
            
            from cgp import cvodeint
            class Test(cvodeint.Cvodeint, SteadyMixin):
                pass
            test = Test(cvodeint.example_ode.logistic_growth, t=[0, 20], y=0.1)
            t, y, flag = test.eq(last_only=False)
            plt.plot(t, y, '-', t[-1], y[-1], 'o')
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
    
    def next_extremum(self, tmax=None, index=0):
        """
        Integrate to next extremum of self.y[index].
        
        :return tuple t, y, e: Time and state arrays, and e = (t[-1], y[-1]).
        
        ..  plot::
            
            from cgp import cvodeint
            def ode(t, y, ydot, f_data):
                ydot[0] = y[1]
                ydot[1] = - y[0]
            class Test(cvodeint.Cvodeint, SteadyMixin):
                pass
            test = Test(ode, t=[0, 10], y=[1, 1])
            t, y, (te, ye) = test.next_extremum()
            plt.plot(t, y, '-', te, ye, 'o')
        """
        t, y, _flag = self.integrate(t=tmax, nrtfn=1, 
            g_rtfn=self.ydoti(index), assert_flag=cvode.CV_ROOT_RETURN)
        return t, y, (t[-1], y[-1])
    
    def cycle(self, index=0, tmax=None, tol=1e-4, n=None):
        extrema = deque([self.next_extremum(tmax, index)], maxlen=n)
        while True:
            t, y, (te, ye) = tup = self.next_extremum(tmax, index)
            extrema.appendleft(tup)
            for lag, (t_, y_, (te_, ye_)) in enumerate(extrema):
                if lag == 0:
                    continue
                diff = unstruct(ye_) - unstruct(ye)
                if self.weighted_rms(diff) < tol:
                    L = reversed(list(extrema)[:lag])
                    t, y, _ = catrec(*L, globalize_time=False)
                    period = te_ - te
                    return t, y, period
                    # Possbly stats, a la ap_stats

from cgp import cvodeint
class Test(cvodeint.Cvodeint, SteadyMixin):
    pass

def f():
    test = Test(cvodeint.example_ode.logistic_growth, t=[0, 20], y=0.1)
    t, y, flag = test.eq(last_only=False)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    from cgp.cvodeint.namedcvodeint import Namedcvodeint
    from cgp.utils.unstruct import unstruct
    class Test(Namedcvodeint, SteadyMixin):
        pass
    
    test = Test(t=[0, 10000])
    t, y, period = test.cycle(tmax=10000)
    # t, y, (te, ye) = test.next_extremum(tmax=10000)
    
    plt.plot(t, unstruct(y).squeeze(), '-') # , te, ye, 'o')
    plt.show()
    plt.savefig("test.png")