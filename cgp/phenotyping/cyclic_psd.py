# -*- coding: utf-8 -*-
"""
Cyclic cellmodels

Functionality
1. Use module cellmodel for initializing the ode's from the cellml specification 
2. Use module cvodeint for solving the ode's with sundials
3. Custom functions for finding and summarizing limit cycles and steady states
 
"""
from ..physmod.cellmlmodel import Cellmlmodel
from pysundials import cvode
import numpy as np
import logging
import ctypes

# Logger objects keep a dict of some relevant information
keys = "asctime levelname name lineno process message"
logging.basicConfig(level=logging.INFO, format = "%(" + ")s\t%(".join(keys.split()) + ")s")
solvelog = logging.getLogger('cyclic')

class Cyclic(Cellmlmodel):
    """
    Class for Cyclic cellmodel
    """
        
    def cycle(self, y=None, ignore_flags=False):
        """
        Simulate with rootfinding for cycle_state until the second extremal point.
        If the initial condition is an extremal point itself the solution will be one complete cycle. 
        
        Example:
        
        >>> sriram = Cyclic()
        >>> t, Y, stats = sriram.cycle()
        >>> stats
        {'ttb': 80.8202..., 'ttp': 43.5743..., 'peak': 56.8233..., 'bottom': 8.4857...}
        >>> from pylab import *
        >>> plot(t,Y.C_1)               #doctest: +SKIP
        >>> plot(stats['ttp'],stats['peak'],'o',stats['ttb'],stats['bottom'],'o')      #doctest: +SKIP 
        >>> t, Y, stats = sriram.cycle()
        >>> plot(t,Y.C_1)               #doctest: +SKIP
        >>> show()                      #doctest: +SKIP
        
        """
        
        if y is None:
            y = self.y
        self._ReInit_if_required(t=0, y=y)
        
        result = [] # list to keep results from integration subintervals

        # integrate short interval to get away from initial minimum
        result.append(self.integrate(t=[self.t[0]+0.001], y=y, nrtfn=0,  
            assert_flag=cvode.CV_TSTOP_RETURN, ignore_flags=ignore_flags))
                
        # integrate from start to first extremal point
        result.append(self.integrate(t=[self.t[0]+self.cycle_max], 
            nrtfn=1, g_rtfn=self.vdot(), assert_flag=cvode.CV_ROOT_RETURN, 
            ignore_flags=ignore_flags))
        tj, yj, flagj = result[-1]

        #integrate from first to second extremal point
        result.append(self.integrate(t=[self.t[0]+self.cycle_max], 
            nrtfn=1, g_rtfn=self.vdot(), assert_flag=cvode.CV_ROOT_RETURN, 
            ignore_flags=ignore_flags))
        tj, yj, flagj = result[-1]
        
        #turn of rootfinding
        self.RootInit(nrtfn=0)
        
        ## drop intervals where flag is None; those were past t_stop
        result = [res for res in result if res[-1] is not None]
        t, Y, flag = zip(*result) # (t, Y, flag), where each is a tuple
        
        ## test for order of peak and bottom and return statistics
        stats = dict()
        if Y[1][-1][0][0]<Y[2][-1][0][0]: #bottom first
            stats['ttp'] = t[2][-1] 
            stats['peak']   = Y[2][-1][0][0]
            stats['ttb'] = t[1][-1] 
            stats['bottom']   = Y[1][-1][0][0]
        else:
            stats['ttp'] = t[1][-1] 
            stats['peak']   = Y[1][-1][0][0]
            stats['ttb'] = t[2][-1] 
            stats['bottom']   = Y[2][-1][0][0]

        # concatenation converts recarray to ndarray, so need to convert back
        Y = np.concatenate(Y).view(np.recarray)
        t = np.concatenate(t)

        return t, Y, stats

    def next_extremum(self, y=None, ignore_flags=False):
        """
        Simulate until next extremal point.
        
        Example:
        
        >>> sriram = Cyclic()
        >>> t, Y, stats = sriram.next_extremum()
        >>> stats
        'peak'
        """
        
        if y is None:
            y = self.y
        self._ReInit_if_required(t=0, y=y)
        
        result = [] # list to keep results from integration subintervals

        # integrate short interval to get away from initial minimum
        result.append(self.integrate(t=[self.t[0]+0.001], y=y, nrtfn=0,  
            assert_flag=cvode.CV_TSTOP_RETURN, ignore_flags=ignore_flags))
        
        # integrate from start to first extremal point
        result.append(self.integrate(t=[self.t[0]+self.cycle_max], 
            nrtfn=1, g_rtfn=self.vdot(), assert_flag=cvode.CV_ROOT_RETURN, 
            ignore_flags=ignore_flags))
        tj, yj, flagj = result[-1]
        
        #turn of rootfinding
        self.RootInit(nrtfn=0)
        
        t, Y, flag = zip(*result) # (t, Y, flag), where each is a tuple
        
        # concatenation converts recarray to ndarray, so need to convert back
        Y = np.concatenate(Y).view(np.recarray)
        t = np.concatenate(t)

        if(Y.view(float)[:,self.cycle_state].max()==Y[-1][0][self.cycle_state]):
            stats = 'peak'
        else:
            stats = 'bottom'
    
        return t, Y, stats  
    
    def vdot(self):
        """
        Return rate-of-change as a function of (t, y, gout, g_data).
        
        """
        def result(t, y, gout, g_data):
            """
            Rate-of-change. Use with CVodeRootInit to find MB peak.
            """
            self.model.ode(t, y, self.model.ydot, None)
            gout[0] = self.model.ydot[self.cycle_state]
            return 0
        return result

    def limit_cycle(self, parameters=[], timecourse=False):
        """
        Solve cvodeint model for stable limit cycle and returns oscillation period, extremal values and timespans between extrema
        
        The function takes in names and values of parameter to be changed from the default
        and return timecourse (optional) and phenotype records for stable cycle.

        Input:
            parameters:      recarray with non-default parameter values
            timecourse:      Booean flag indicating whether the par2pheno should return timecourse data in addition to aggregated phenotypes
            
        Output: 
            phenotypes:     phenotype recarray with phenotype names as dtype.names and
            t (optional):   array with integration timesteps
            Y (optional):   recarray with state variables for times in t
        
        >>> leloup = Leloup()
        >>> phenotypes = leloup.limit_cycle()                            
        >>> phenotypes.period
        array([ 24.16857...])
        """
        
        #save initial y and parameters
        pr0 = self.pr.copy()
        solvelog.info('y (initial): '+str(self.y))
                      
        #and change parameter values
        if parameters:
            newnames = parameters.dtype.names
            newvalues = parameters[0]
            for i in range(len(newnames)):
                setattr(self.pr,newnames[i],newvalues[i])

        try:

            #solve until cycles have converged
            self._ReInit_if_required(t=[0,0], y=self.y0r)
            convergence = False
            for tstop in [2**N for N in range(6,32)]:
                tint, Yint, flag = self.integrate(t=tstop)
                t, Y, stats = self.next_extremum()
                if stats=='peak': #find first bottom point for MB
                    t, Y, stats = self.next_extremum()
                if stats=='bottom':
                    t0 = np.array(self.tret) # time for second extrema after t=500
                else:
                   Exception('Strange extremal point')
                t,Y,stats = self.cycle()
                t1, Y1, stats = self.cycle()
                t2, Y2, stats = self.cycle()
                
                # convergence test for period
                p1 = t1[-1] - t1[0]
                p2 = t2[-1] - t2[0]
                diff2 = (np.max([p1,p2])-np.min([p1,p2]))/np.mean([p1,p2])
                
                # convergence test for extremal values and times
                ext1 = np.array([cycle_extrema(t1-t1[0],Y1,var) for var in Y1.dtype.names])
                ext2 = np.array([cycle_extrema(t2-t2[0],Y2,var) for var in Y2.dtype.names])
                ext1[0,3]=0.0
                ext2[0,3]=0.0
                diffmax = np.max(np.abs(ext1[:,0]-ext2[:,0])/((ext1[:,0]+ext2[:,0])/2))
                diffmin = np.max(np.abs(ext1[:,2]-ext2[:,2])/((ext1[:,2]+ext2[:,2])/2))
                diffmaxt = np.max(np.abs(ext1[:,1]-ext2[:,1]))/np.mean([p1,p2])
                diffmint = np.max(np.abs(ext1[:,3]-ext2[:,3]))/np.mean([p1,p2])
                solvelog.info("time %07d per %f maxt %f mint %f max %f min %f" % (tint[-1],diff2,diffmaxt,diffmint,diffmax,diffmin))
                
                maxdiff = 1e-4
                if diff2<maxdiff and diffmaxt<maxdiff and diffmint<maxdiff and diffmax<maxdiff and diffmin<maxdiff:
                    convergence = True
                    break 
                    
            if not convergence:
                solvelog.error('Solution did not converge to limit cycle - raising exception')
                solvelog.error('Synchrony metric ( 0.001 needed )'+str(diff1))
                solvelog.error('Period metric    ( 1e-4 needed )'+str(diff2))
                raise Exception('Solution did not converge to limit cycle')

            #after convergence solve for one period and extract phenotypes
            t, Y, stats = self.cycle()
        
        finally: 
            #reset parameters and inital condition 
            self.pr[:] = pr0
        
        phenotype_names = ['period']
        period = t[-1]-t[0]
        phenotype_values = [period]
        phenotype_names.append('tconv')
        phenotype_values.append(tstop)
        extrema = [ (var,cycle_extrema(t-t[0],Y,var)) for var in Y.dtype.names ]
        
        for var in extrema:
            phenotype_names.append(var[0]+'_bottom')
            phenotype_names.append(var[0]+'_peak')
            phenotype_names.append(var[0]+'_ttb') #time from MP bottom to var[0].bottom
            phenotype_names.append(var[0]+'_ttp') #time from var[0].bottom to var[0].peak 
            phenotype_values.append(var[1][2])
            phenotype_values.append(var[1][0])
            phenotype_values.append(var[1][3])
            if(var[1][3]<var[1][1]):    #peak after bottom
                phenotype_values.append(var[1][1]-var[1][3])
            else:
                phenotype_values.append(period-var[1][3]+var[1][1])
            
        #create phenotype recarray
        dtype = [(name, float) for name in phenotype_names]
        phenotypes = np.array(phenotype_values)
        phenotypes = phenotypes.view(dtype,np.recarray)

        #return data
        if timecourse:
            return (t,Y,phenotypes)
        else :
            return phenotypes
            
    def steady_state(self, parameters=[],timecourse=False):
        """
        Solve cvodeint model for stable steady state

        Example: ( Hopf bifurcation for vmB~=0.03 see Table 2 in paper)
        
        >>> leloup = Leloup()
        >>> leloup.pr.vmB = 0.001
        >>> t, Y, stats = leloup.integrate(t=1000)
        >>> from pylab import *
        >>> plot(t,Y.MB,t,Y.MC,t,Y.MP)              #doctest: +SKIP
        >>> legend(['MB','MC','MP'])                #doctest: +SKIP
        >>> show()                                  #doctest: +SKIP
        >>> ss = leloup.steady_state()
        >>> [ss.period, ss.MB_peak, ss.MB_bottom ]
        [array([ inf]), array([ 1.73769777]), array([ 1.73769777])]
        """

        #save initial parameter values
        pr0 = self.pr.copy()

        #change parameter values
        if parameters:
            newnames = parameters.dtype.names
            newvalues = parameters[0]
            for i in range(len(newnames)):
                setattr(self.pr,newnames[i],newvalues[i])

        #solve until states have converged
        convergence = False
        try:
            for tstop in [4**N for N in range(3,16)]:
                #print 'cGP_leloup:101 Integrating to t=',tstop
                t, Y, flag = self.integrate(t=[max(self.t)+tstop])
                diff1 = (Y.view(float).max(axis=0)-Y.view(float).min(axis=0)).max()
                solvelog.debug('cGP_leloup.py:227 Max state difference:'+str(diff1))
                if diff1<1e-8:   
                    convergence=True
                    break

            if not convergence:
                solvelog.error('Solution did not converge to steady state - raising exception')
                solvelog.error('Max state difference ( 1e-8 needed )'+str(diff1))
                raise Exception('Solution did not converge to steady state')
        
        finally:    #reset parametr values
            self.pr[:] = pr0
        
        phenotype_names = ['period']
        phenotype_values = [np.inf]
        phenotype_names.append('tconv')
        phenotype_values.append(tstop)

        extrema = [ (var,Y[var][0],Y[var][0],np.inf,np.inf) for var in Y.dtype.names ]
        
        for var in extrema:
            phenotype_names.append(var[0]+'_bottom')
            phenotype_names.append(var[0]+'_peak')
            phenotype_names.append(var[0]+'_ttb') #time from MP bottom to var[0].bottom
            phenotype_names.append(var[0]+'_ttp') #time from var[0].bottom to var[0].peak 
            phenotype_values.append(var[1])
            phenotype_values.append(var[2])
            phenotype_values.append(var[3])
            phenotype_values.append(var[4])

        #create phenotype recarray
        dtype = [(name, float) for name in phenotype_names]
        phenotypes = np.array(phenotype_values)
        phenotypes = phenotypes.view(dtype,np.recarray)

        #return data
        if timecourse:
            return (t,Y,phenotypes)
        else :
            return phenotypes
        
    def __init__(self, exposure_workspace=
        "500b72e21302febd0f3dc746dc22af81/sriram_bernot_kepes_2007",
        cycle_max=200, cycle_state=0, chunksize=10000, maxsteps=1e6, **kwargs):
        """
        Return a Cyclic model object
        
        "modelname" is the model name at cellml.org, see ?Cellmlc2py.
        Other keyword arguments are passed through to Cellmlmodel().
        
        This constructor sets stim_start = 0.0, overriding the default.
        
        >>> sriram = Cyclic()
        """
        super(Cyclic, self).__init__(exposure_workspace, **kwargs)
        self.cycle_max = cycle_max      # max length of cycle
        self.cycle_state = cycle_state  # index determining which state variable to use for rootfinding
        self.chunksize = chunksize      # cvode chunksize
        self.maxsteps = maxsteps        # cvode maxsteps
  
        
class Leloup(Cyclic):
    '''
    Leloup and Goldbeter 2004, Mammalian circadian rhytm model ( doi:1016/j.jtbi.2004.04.040 )
    CellML model at http://models.cellml.org/workspace/leloup_goldbeter_2004
    '''
    
    def __init__(self, exposure_workspace="975e54d2d30fc68ebe488375a0731cb2/leloup_goldbeter_2004", **kwargs):
        """
        Return a Leloup model object
        
        "modelname" is the model name at cellml.org, see ?Cellmlc2py.
        Other keyword arguments are passed through to Cellmlmodel().
          
        >>> leloup = Leloup()
        """
        super(Leloup, self).__init__(exposure_workspace, **kwargs)
        self.abstol = ctypes.c_double(1e-10)
        self.reltol = reltol = 1e-10
        
class Sriram(Cyclic):
    '''
    Sriram, Bernot and Kepes 2007, Yeast cell cycle model ( doi:10.1049/iet-syb:20070018 )
    CellML model at http://models.cellml.org/workspace/sriram_bernot_kepes_2007
    
    Example:
    
    >>> sri = Sriram()
    >>> t, Y, stats = sri.integrate(t=[0,1000])
    >>> from pylab import *
    >>> plot(t,Y.T_1)                           #doctest: +SKIP
    >>> phenotypes = sri.limit_cycle()
    >>> phenotypes.period
    array([ 100.34604991])
    
    Hopf bifurcation for v_d2~=40 (Fig 5 in paper)
    
    >>> sri.pr.v_d2 = 50
    >>> t, Y, stats = sri.integrate(t=[0,1000])
    >>> plot(t,Y.T_1)                           #doctest: +SKIP
    >>> xlabel('time'); ylabel('T_1'); legend(['v_d2=1.052','v_d2=50.0'])   #doctest: +SKIP
    >>> phenotypes = sri.steady_state()
    >>> phenotypes.period
    array([ inf])
    '''
 
    def __init__(self, exposure_workspace=
        "500b72e21302febd0f3dc746dc22af81/sriram_bernot_kepes_2007", **kwargs):
        """
        Return a Sriram model object
                  
        >>> sriram = Sriram()
        """
        super(Sriram, self).__init__(exposure_workspace, **kwargs)
        
class Qu(Cyclic):
    '''
    Qu, MacLellan and Weiss 2003. Eukaryot cell cycle model ( http://www.ncbi.nlm.nih.gov/pubmed/14645053 )
    CellML model at http://models.cellml.org/workspace/qu_maclellan_weiss_2003
    
    TODO: check limit cycle
    
    Example:
    
    >>> qu = Qu()
    >>> t, Y, stats = qu.integrate(t=500)
    >>> t, Y, stats = qu.integrate(t=750)
    
    Figure 6 B in paper
    
    >>> from pylab import *
    >>> plot(t,Y.cyclin_CDK_active,t,Y.cyclin_CDK_inactive,t,Y.cyclin_CDK_active+Y.cyclin_CDK_inactive+Y.cyclin)        #doctest: +SKIP
    '''
 
    def __init__(self, exposure_workspace=
        "119bacd67eb14ac06448dedf40b96ab3/qu_maclellan_weiss_2003", **kwargs):
        """
        Return a Qu model object
                  
        >>> qu = Qu()
        """
        super(Qu, self).__init__(exposure_workspace, **kwargs)


def cycle_extrema(t,Y,varname):
    """
    Take in results from cycle() and return time and value for extrema
    for varname.
    
    Output: [maxvalue, maxtime, minvalue, mintime]
    
    Trigonometry example:
    
    >>> t = np.arange(0,2*np.pi,0.001)
    >>> Y = np.rec.fromarrays([np.sin(t),np.cos(t)],names='sin,cos')
    >>> cycle_extrema(t,Y,'sin')
    [0.99999997925861284, 1.571, -0.99999992434713114, 4.7119999999999997]
    >>> cycle_extrema(t,Y,'cos')
    [1.0, 0.0, -0.99999991703445223, 3.1419999999999999]
    """
    maxvalue = Y[varname].max()
    maxfind = Y[varname]==maxvalue
    maxtime = t[maxfind.flatten()]
    if(len(maxtime)>1):
        if(np.max(np.diff(maxtime))>0):
            Exception('Multiple maxima')
    maxtime = maxtime[0]

    minvalue = Y[varname].min()
    minfind = Y[varname]==minvalue
    mintime = t[minfind.flatten()]
    if(len(mintime)>1):
        if(np.max(np.diff(mintime))>0):
            Exception('Multiple maxima')
    mintime=mintime[0]
    
    return [maxvalue,maxtime,minvalue,mintime]

def interpolate(t,Y,N_interp):
    import scipy
    t_interp = scipy.linspace(t[0],t[-1],N_interp) #interpolation timepoints
    phenos = Y.dtype.names #phenotype names
    x_interp = np.empty(shape=(N_interp,len(phenos))) #interpolated solution
    for j in range(len(phenos)):
        x = Y[phenos[j]] #solution from pysundials
        x.shape = t.shape
        x_interp[:,j]=scipy.interp(t_interp,t,x)
    return (t_interp,x_interp)

def normalize(Y):
    y = Y.view(float)
    y = y-y.min(axis=0)
    y = y/y.max(axis=0)
    return(y)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
