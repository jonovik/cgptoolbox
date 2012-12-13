"""Find steady state or stable limit cycle of dynamic system."""

from collections import deque

import numpy as np
from pysundials import cvode

from cgp.virtexp.elphys.clampable import catrec
from cgp.utils.unstruct import unstruct

class AttractorMixin(object):
    """Mixin to find steady state or stable limit cycle of dynamic system."""
    # Integrate until norm of rate-of-change < tol (steady state) or 
    # norm of difference between successive extrema < tol (stable limit cycle))
    
    def weighted_rms(self, y):
        """Norm of vector using CVODE weights based on reltol and abstol."""
        weights = 1.0 / (self.reltol * np.abs(y.view(float)) + self.abstol)
        return ((weights * y ** 2).sum() / weights.sum()) ** 0.5
    
    def ydotnorm(self, tol):
        """
        Make rootfinding function: :func:`weighted_rms` norm of rate of change.
        
        See cvode.h and associated CVODE help pages.
        """
        ydot = np.empty_like(self.y)
        
        def result(t, y, gout, g_data=None):
            """Rootfinding function for convergence to equilibrium."""
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
        
        ..  plot::
            :include-source:
            
            >>> from matplotlib import pyplot as plt
            >>> from cgp import cvodeint
            >>> from cgp.phenotyping.attractor import AttractorMixin
            >>> class Test(cvodeint.Cvodeint, AttractorMixin):
            ...     pass
            >>> test = Test(cvodeint.example_ode.logistic_growth, 
            ...     t=[0, 20], y=0.1)
            >>> t, y, flag = test.eq(last_only=False)
            >>> h = plt.plot(t, y, '-', t[-1], y[-1], 'o')
        """
        g_rtfn = self.ydotnorm(tol)
        gout = np.zeros(1)
        # CV_ROOT_RETURN may never happen if already converged, so check now
        g_rtfn(self.tret.value, self.y, gout)
        if gout < 0:
            t = np.atleast_1d(self.tret)
            y = np.copy(self.y)
            flag = cvode.CV_ROOT_RETURN
        else:
            if tmax is None:
                tmax = self.t[-1]
            t, y, flag = self.integrate(t=tmax, nrtfn=1, g_rtfn=g_rtfn, 
                assert_flag=cvode.CV_ROOT_RETURN)
        y = y.squeeze()
        if last_only:
            return t[-1], y[-1]
        else:
            return t, y, flag
    
    def next_extremum(self, tmax=None, index=0):
        """
        Integrate to next extremum of self.y[index].
        
        :return tuple t, y, e: Time and state arrays, and e = (t[-1], y[-1]).
        
        ..  plot::
            :include-source:
            
            >>> from matplotlib import pyplot as plt
            >>> from cgp import cvodeint
            >>> from cgp.phenotyping.attractor import AttractorMixin
            >>> def ode(t, y, ydot, f_data):
            ...     ydot[0] = y[1]
            ...     ydot[1] = - y[0]
            >>> class Test(cvodeint.Cvodeint, AttractorMixin):
            ...     pass
            >>> test = Test(ode, t=[0, 10], y=[1, 1])
            >>> t1, y1, (te1, ye1) = test.next_extremum()
            >>> np.testing.assert_almost_equal(te1, np.pi / 4)
            >>> np.testing.assert_allclose(ye1, [np.sqrt(2), 0], atol=1e-6)
            >>> t2, y2, (te2, ye2) = test.next_extremum()
            >>> h = plt.plot(t1, y1, '-', t2, y2, ':', 
            ...     te1, ye1[0], 'o', te2, ye2[0], 's')
        """
        t, y, _flag = self.integrate(t=tmax, nrtfn=1, 
            g_rtfn=self.ydoti(index), assert_flag=cvode.CV_ROOT_RETURN)
        return t, y.squeeze(), (t[-1], y[-1])
    
    def cycle(self, index=0, tmax=None, tol=1e-4, n=None):
        """
        Find limit cycle (integrate until successive extrema are almost equal).
        
        :param str_or_int index: Index of state variable to base search on.
        :param float tmax: Time limit for aborting search for limit cycle.
        :param float tol: Tolerance for weighted root-mean-square norm of 
            change in state vector between successive extrema.
        :param int n: Keep history of up to n extrema while searching for 
            limit cycle.
        
        ..  plot::
            :include-source:
            
            >>> from matplotlib import pyplot as plt
            >>> from cgp.cvodeint.namedcvodeint import Namedcvodeint
            >>> from cgp.utils.unstruct import unstruct
            >>> from cgp.phenotyping.attractor import AttractorMixin
            >>> class Test(Namedcvodeint, AttractorMixin):
            ...     '''Inherits van der Pol equations as default example.'''
            ...     pass
            >>> test = Test(t=[0, 10000])
            >>> t, y, period = test.cycle()
            >>> period
            6.663...
            >>> h = plt.plot(t, unstruct(y), '-')
        """
        if tmax is None:
            tmax = self.t[-1]
        extrema = deque([self.next_extremum(tmax, index)], maxlen=n)
        while True:
            t, y, (te, ye) = tup = self.next_extremum(tmax, index)
            extrema.appendleft(tup)
            # pylint:disable=W0612,C0301
            for lag, (t_, y_, (te_, ye_)) in enumerate(extrema): #@UnusedVariable
                if lag == 0:
                    continue
                diff = unstruct(ye_) - unstruct(ye)
                if self.weighted_rms(diff) < tol:
                    L = reversed(list(extrema)[:lag])
                    t, y, _ = catrec(*L, globalize_time=False)
                    period = te - te_
                    return t, y.squeeze(), period

# TODO: Separate function to compute statistics from raw trajectory
