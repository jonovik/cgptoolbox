"""Mockup model of action potential: :class:`Test_cell`."""
# pylint: disable=C0111,E1002,W0612

import numpy as np

from ..cvodeint.namedcvodeint import Namedcvodeint
from ..virtexp.elphys import Paceable, Clampable
from ..utils.rec2dict import dict2rec

class Test_cell(Namedcvodeint, Paceable, Clampable):
    """
    Mockup of action potential model, see :meth:`f_ode` and :meth:`solution`.
    
    ..  inheritance-diagram:: 
        cgp.cvodeint.core.Cvodeint
        cgp.cvodeint.namedcvodeint.Namedcvodeint
        cgp.virtexp.elphys.Paceable
        cgp.virtexp.elphys.Clampable
        Test_cell
        :parts: 1
    
    ..  plot::
        :width: 400
        
        from cgp.test.test_elphys_analytic import Test_cell
        c = Test_cell()
        for t, y, stats in c.aps(n=2):
            plt.plot(t, y.V, '.-')
    """
    
    # Define data type and default values for parameters
    pr = dict2rec(m=np.log(20), stim_duration=1.0, stim_amplitude=10.0, 
                  stim_period=2.0).view(np.recarray)
    
    def __init__(self):
        super(Test_cell, self).__init__(
            self.f_ode, t=(0, 2), y=dict2rec(V=1.0), p=self.pr, reltol=1e-10)
    
    def f_ode(self, t, y, ydot, f_data=None):  # pylint: disable=E0202
        """ODE for Test_cell with linear upstroke and exponential decay."""
        if (t % self.pr.stim_period) < self.pr.stim_duration:
            ydot[:] = self.pr.stim_amplitude
        else:
            ydot[:] = - self.pr.m * y
    
    def solution(self, t, y):
        """Solved ODE for Test_cell: linear upstroke and exponential decay."""
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
    
    def test_ap_reset(self):
        """Verify that time resets to 0 after calls to :meth:`ap`."""
        with self.autorestore():
            t0, _y0, _stats0 = self.ap()
            t1, _y1, _stats0 = self.ap()
        assert t0[0] == t1[0] == 0
    
    def test_aps(self):
        """Test the :meth:`aps` generator."""
        with self.autorestore():
            (t0, _y0, stats0), (t1, _y1, stats1) = self.aps(n=2)
        assert t0[-1] == t1[0] == self.pr.stim_period
        assert stats0["ttp"] == stats1["ttp"] == self.pr.stim_duration
