"""Mockup model of action potential."""
# pylint: disable=C0111,E1002

import numpy as np

from ..cvodeint.namedcvodeint import Namedcvodeint
from ..virtexp.elphys import Paceable
from ..utils.rec2dict import dict2rec

class Test_cell(Namedcvodeint, Paceable):
    
    pr = dict2rec(m=3, stim_duration=1.0, stim_amplitude=10.0, 
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
        np.testing.assert_allclose(y.V.squeeze()[-1], ys[-1])
    
    def __init__(self):
        super(Test_cell, self).__init__(
            self.f_ode, t=(0, 2), y=dict2rec(V=1.0), p=self.pr, reltol=1e-10)
