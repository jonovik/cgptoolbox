"""Mockup model of action potential."""
# pylint: disable=C0111,E1002

import numpy as np

from ..cvodeint.namedcvodeint import Namedcvodeint
from ..virtexp.elphys import Paceable, Clampable
from ..utils.rec2dict import dict2rec

class Test_cell(Namedcvodeint, Paceable, Clampable):
    
    pr = dict2rec(m=np.log(20), stim_duration=1.0, stim_amplitude=10.0, 
                  stim_period=2.0).view(np.recarray)
    
    def f_ode(self, t, y, ydot, f_data=None):
        if (t % self.pr.stim_period) < self.pr.stim_duration:
            ydot[:] = self.pr.stim_amplitude
        else:
            ydot[:] = - self.pr.m * y
    
    def solution(self, t, y):
        assert (t <= self.pr.stim_period).all()
        ystart = y[0].view(float)
        t0 = t[t <= self.pr.stim_duration]
        t1 = t[t > self.pr.stim_duration]
        y0 = ystart + t0 * self.pr.stim_amplitude
        y1 = y0[-1] * np.exp(- self.pr.m * (t1 - t0[-1]))
        return np.r_[y0, y1]
    
    def test_ap_solution(self):
        """Test that integration works up to the next stimulus."""
        with self.autorestore():
            t, y, _stats = self.ap()
        ys = self.solution(t, y)
        np.testing.assert_allclose(y.V.squeeze()[:-1], ys[:-1])
    
    def test_ap_solution_end(self):
        """Test if the next stimulus "spills over" to the previous interval."""
        with self.autorestore():
            t, y, _stats = self.ap()
        ys = self.solution(t, y)
        np.testing.assert_allclose(y.V.squeeze()[-1], ys[-1], rtol=1e-6)
    
    def __init__(self):
        super(Test_cell, self).__init__(
            self.f_ode, t=(0, 2), y=dict2rec(V=1.0), p=self.pr, reltol=1e-10)
    
    def test_ap(self):
        """
        Test ported from :mod:`cgp.virtexp.elphys`.
        
        >>> cell = Test_cell()
        >>> t, y, stats = cell.ap()
        
        Calling :meth:`ap` again resumes from the previous state, resetting 
        time to 0:
        
        >>> t1, Y1, stats1 = cell.ap()
        >>> t1[0]
        0.0
        
        Parameters governing the stimulus setup:
        
        >>> [(s, cell.pr[s]) for s in cell.pr.dtype.names
        ...     if s.startswith("stim_")]
        [('stim_duration', array([ 1.])), 
         ('stim_amplitude', array([ 10.])), 
         ('stim_period', array([ 2.]))]
        
        >>> from pprint import pprint # platform-independent order of dict items
        >>> pprint(stats)
        {'amp': 9.999...,
         'base': 1.0,
         'decayrate': array([ 4.389...
         'peak': 10.999...,
         't_repol': array([  1.086...,   1.202...,  1.382...,  1.569...]),
         'ttp': 1.0}
        
        To temporarily modify initial state or parameters, use 
        :meth:`~cvodeint.namedcvodeint.Namedcvodeint.autorestore`.
        
        >>> with cell.autorestore(V=10, stim_period=3.0):
        ...     t, y, stats = cell.ap()
        >>> t[-1]
        3.0
        >>> pprint(stats)
        {'amp': 10.000...,
         'base': 10.0,
         'decayrate': array([ 15.546...
         'peak': 20.000...,
         't_repol': array([ 1.044..., 1.096...,   1.156..., 1.199...]),
         'ttp': 1.0}
        """
        pass
    
    def test_aps(self, n=5, y=None, pr=None, *args, **kwargs):
        """
        Test ported from :mod:`cgp.virtexp.elphys`.
        
        >>> bond = Test_cell()
        >>> aps = list(bond.aps(n=2))
        
        You can iterate over the list of tuples like so:
        
        >>> from pprint import pprint
        >>> for t, y, stats in aps:
        ...     print t[-1], y[-1].V
        ...     pprint(stats)
        2.0 [ 0.55000...]
        {'amp': 9.99999...,
         'base': 1.0,
         'decayrate': array([ 4.389...}
        4.0 [ 0.52750...]
        {'amp': 10.0,
         'base': 0.55000...,
         'decayrate': array([ 3.779...}
        
        Separate lists for time, states, stats:
        
        >>> t, y, stats = zip(*aps)
        
        Time is reckoned consecutively, not restarting for each action potential.
        
        >>> t[0][-1] == t[1][0]
        True
        
        Parameters can vary between intervals by specifying *pr* as a list.
        
        >>> p = np.tile(bond.pr, 3)
        >>> p["stim_period"] = 20, 30, 40
        >>> with bond.autorestore():
        ...     [ti[-1] for ti, yi, statsi in bond.aps(pr=p)]
        [20.0, 50.0, 90.0]
        
        In case of a :exc:`~cvodeint.core.CvodeException`, the exception 
        instance has a *result* attribute with the results thus far: 
        A list of *(t, y, stats)*, where *stats* is *None* for the failed action 
        potential.
                
        (Doctests to guard against accidental changes.)
        
        >>> ix = np.r_[0:2, -2:0]
        >>> t[1][ix]
        array([ 2.        ,  2.0000... ,  3.99999...,  4.        ])
        >>> y[1].V[ix]
        array([[ 0.55000...], [ 0.55003...], [ 0.52750...], [ 0.52750...]])
        """
        pass
    
    def test_dynclamp(self):
        stim_amplitude = float(self.pr.stim_amplitude)
        assert stim_amplitude != 0
        with self.dynclamp(-140):
            assert self.pr.stim_amplitude == 0
        np.testing.assert_equal(self.pr.stim_amplitude, stim_amplitude)
