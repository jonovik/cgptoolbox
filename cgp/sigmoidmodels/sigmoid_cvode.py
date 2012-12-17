"""
Solve sigmoid ode model equations describing diploid gene regulatory networks.

The class Sigmoidmodel inherits from Namedcvodeint and contains an adjacency
matrix defining the connectivity of the network, a function defining the right
hand side of the ODEs and a function for sampling heritable variation in allelic
parameter values for cGP studies. The parameter ranges default to the ranges
used in Gjuvsland et al. (2007).

Reference:

* Gjuvsland AB, Hayes B, Omholt SW, Carlborg O (2007)
  `Statistical Epistasis Is a Generic Feature of Gene Regulatory Networks
  <http://www.genetics.org/content/175/1/411.full>`_ Genetics 175(1): 411-420.
  
"""

from __future__ import division
from ..cvodeint.namedcvodeint import Namedcvodeint
from ..sigmoidmodels.doseresponse import Hill, R_logic
import numpy as np

class Sigmoidmodel(Namedcvodeint):
    """
    Class to solve sigmoid ode model equations describing diploid gene regulatory networks.
    
    FIXME: Inherits from Namedcvodeint, but the default y is not a recarray.
    
    Constructor arguments *t*, *y* and *p* are as for :class:`~cgp.cvodeint.namedcvodeint.Namedcvodeint`.
    
    :param array_like p: Diploid parameters in the for of arecarray with 18 fields where each field contains 
        an array with two elements corresponding to the two alleles. The parameter specify the following for the 
        three genes: maximal production rate (*alpha*) and decay rate (*gamma*) as well :meth:`~cgp.sigmoidmodels.doseresponse.Hill` 
        function parameters treshold (*theta*) and regulation steepness (*p*) for two (potential) regulators
    :param array_like y: Diploid state vector representing the expression level
        
    :param array_like adjmatrix: 3x3 `adjacency matrix <http://en.wikipedia.org/wiki/Adjacency_matrix>`_
        describing signed directed graph of the regulatory system with genes 1-3. adjmatrix[i,j] describes 
        the regulatory effect of gene j+1 on the production of gene i+1. 0 means no effect, 1 means activation
        and -1 means repression.
    :param list boolfunc: a list of 3 integers in the range 1-16 corresponding to the boolean function indexes in
        :meth:`~cgp.sigmoidmodels.doseresponse.R_logic`.
    
    >>> s = Sigmoidmodel()
    
    Print outputs a textual description of network connectivity and regulation modes
    
    >>> print s
    Instance of class Sigmoidmodel() representing a 3-gene regulatory network where:
    * Gene 1 has no regulators and production is always on.
    * Gene 2 has one activator of its production, gene 1.
    * Gene 3 has one activator of its production, gene 2.
    
    If we start with zero expression in this network only gene 1, which is always on, increases
    
    >>> y = np.zeros(6)
    >>> ydot = np.zeros(6)
    >>> s.equations_diploid_adjacency(s.t0, y, ydot, None)
    >>> ydot
    array([ 150.,  150.,    0.,    0.,    0.,    0.])
    
    Once there is a certain amount of gene 1 then gene 2 starts to increase
    
    >>> y = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    >>> s.equations_diploid_adjacency(s.t0,y,ydot,None)
    >>> ydot
    array([ 100. ,  100. ,    3.24...,    3.24...,    0. ,    0. ])
    
    But seriously, we can't continue looking at derivatives, let's integrate!
    
    .. plot::
        :width: 400
        :include-source:

        from matplotlib import pyplot as plt
        from cgp.sigmoidmodels.sigmoid_cvode import Sigmoidmodel
        
        s = Sigmoidmodel()
        t,y,stats = s.integrate()
        plt.plot(t,y[:,0]+y[:,1],t,y[:,2]+y[:,3],t,y[:,4]+y[:,5],)
        plt.xlabel('t')
        plt.ylabel('Expression level')
        plt.legend(('Gene 1','Gene 2','Gene 3'))
    """
    
    #: Adjacency matrix defining network connectivity
    adjmatrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    
    #: ODE parameters (production rates, decay rates, 
    #: shape of gene regulation function:
    _origpar = [ 
        ('alpha1',  150, 100),
        ('theta11', 20,  20),
        ('p11',     5.5, 9),
        ('theta12', 20,  20),
        ('p12',     5.5, 9),
        ('gamma1',  10,  0),
        ('alpha2',  150, 100),
        ('theta21', 20,  20),
        ('p21',     5.5, 9),
        ('theta22', 20,  20),
        ('p22',     5.5, 9),
        ('gamma2',  10,  0),
        ('alpha3',  150, 100),
        ('theta31', 20,  20),
        ('p31',     5.5, 9),
        ('theta32', 20,  20),
        ('p32',     5.5, 9),
        ('gamma3',  10,  0)
        ]

    #: Names of parameters, cf. Gjuvsland et al. (2007)
    par_names, par_mean, par_span = zip(*_origpar)
    #: Mean parameter values
    par_mean = np.array([[_i, _i] for _i in par_mean])
    #: Span in uniformly sampled parameter values
    par_span = np.array([[_i, _i] for _i in par_span])
    
    _dtype = np.dtype([(_name, float, (2,)) for _name in par_names])
    p = par_mean.flatten().view(_dtype, np.recarray)

    def __init__(self, t=(0, 1), y=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
        p=p, adjmatrix=np.array([[0,0,0],[1,0,0],[0,1,0]]), boolfunc=None):
        """
        Constructor arguments *t*, *y* and *p* are as for :class:`~cgp.cvodeint.namedcvodeint.Namedcvodeint`.
        
        :param array_like adjmatrix: 3x3 `adjacency matrix <http://en.wikipedia.org/wiki/Adjacency_matrix>`_
            describing signed directed graph of the regulatory system with genes 1-3. adjmatrix[i,j] describes 
            the regulatory effect of gene j+1 on the production of gene i+1. 0 means no effect, 1 means activation
            and -1 means repression.
        :param list boolfunc: a list of 3 integers in the range 1-16 corresponding to the boolean function indexes in
            :meth:`~cgp.sigmoidmodels.doseresponse.R_logic`.

        >>> s = Sigmoidmodel()
        >>> s.boolfunc
        array([ 16., 5., 5.])
                
        # gene 1 and 2 both activate gene 3, boolean combination of regulators default to AND
        >>> s = Sigmoidmodel(adjmatrix=np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]]))
        >>> s.boolfunc
        array([ 16., 16., 1.])
  
        # gene 1 and 2 both activate gene 3, specify combination of regulators to be OR
        >>> s = Sigmoidmodel(adjmatrix=np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]]), boolfunc=[16, 16, 9])
        
        # incompatible adjmatrix and boolfunc raises Exception 
        >>> s = Sigmoidmodel(adjmatrix=np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]]), boolfunc = [16, 16, 5])
        Traceback (most recent call last):
            ...
        Exception: Mismatch: adjmatrix[2, :] indicates 2 regulator(s), but boolfunc[2] does not.
        """
        t = np.array(t)
        y = np.asanyarray(y)
        if boolfunc == None:
            self.boolfunc = self.adjacency_to_boolfunc(adjmatrix)
        else:
            self.check_adjacency_and_boolfunc(adjmatrix, boolfunc)
            self.boolfunc = boolfunc
        self.adjmatrix = adjmatrix
        super(Sigmoidmodel, self).__init__(self.equations_diploid_adjacency, 
            t, y, p=p)
            
    def __str__(self):
        """
        Textual description of the connectivity and mode of regulation.
        
        :TODO: make a corresponding graphical representation using networkx
        
        >>> s = Sigmoidmodel()
        >>> print s
        Instance of class Sigmoidmodel() representing a 3-gene regulatory 
        network where:
        * Gene 1 has no regulators and production is always on.
        * Gene 2 has one activator of its production, gene 1.
        * Gene 3 has one activator of its production, gene 2.
        
        >>> s = Sigmoidmodel(adjmatrix=np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]]), boolfunc=[16, 16, 9])
        >>> print s
        Instance of class Sigmoidmodel() representing a 3-gene regulatory network where:
        * Gene 1 has no regulators and production is always on.
        * Gene 2 has no regulators and production is always on.
        * Gene 3 has two regulators of production. Gene 1 activates, gene 2 activates. The boolean function 
        with index 9 in cgp.sigmoidmodels.doseresponse.R_logic is used to combine the regulatory signals.

        """
        self.check_adjacency_and_boolfunc(self.adjmatrix, self.boolfunc)
        text = "Instance of class Sigmoidmodel() representing a 3-gene "
        text += "regulatory network where:"

        reglist = self.adjacency_to_reglist(self.adjmatrix)

        for i in range(3):
            regmode = self.adjmatrix[i, :].nonzero()[0]
            if np.shape(regmode) == (0,):
                # adjmatrix indicates no regulators
                text += "\n  * Gene {} has no regulators ".format(i + 1)
                if self.boolfunc[i] == 15:
                    text += "and production is always off."
                if self.boolfunc[i] == 16:
                    text += "and production is always on."
            elif np.shape(regmode) == (1,):
                # adjmatrix indicates one regulator
                text += "\n  * Gene {} has one".format(i + 1)
                if self.adjmatrix[i, regmode[0]] == 1:
                    text += " activator"
                else:
                    text += " inhibitor"
                text += " of its production, "
                text += "gene {}.".format(reglist[i][0] + 1)
            else:
                text += "\n  * Gene {} has two regulators ".format(i + 1)
                text += "of production. Gene {}".format(reglist[i][0] + 1)
                if self.adjmatrix[i, regmode[0]] == 1:
                    text += " activates"
                else:
                    text += " inhibits"
                text += ", gene " + str(reglist[i][1] + 1)
                if self.adjmatrix[i, regmode[1]] == 1:
                    text += " activates. "
                else:
                    text += " inhibits. "
                text += "The boolean function \n    "
                text += "with index {} ".format(self.boolfunc[i])
                text += "in cgp.sigmoidmodels.doseresponse.R_logic "
                text += "is used to combine the regulatory signals."
        
        return text
    
    def equations_diploid_adjacency(self, t, y, ydot, fdata):
        """
        ODE rate function for a diploid model with three loci X1, X2, X3.
        
        This is a general model where each gene can be regulated by two of the 
        three genes. 
        The actual motif is controlled by the adjacency matrix adjmatrix.

        >>> s = Sigmoidmodel()
        >>> ydot = np.zeros(6)
        >>> s.equations_diploid_adjacency(s.t0, s.y, ydot, None)
        >>> ydot
        array([ 150.,  150.,    0.,    0.,    0.,    0.])
        """
        
        Y = np.nan*np.ones((4,))
        Y[0] = y[0] +y[1]    # sum of x11 and x12
        Y[1] = y[2] + y[3]   # sum of x21 and x22
        Y[2] = y[4] + y[5]   # sum of x31 and x32
        Y[3] = np.nan        # dummy variable used when there is no regulator
        
        reglist = self.adjacency_to_reglist(self.adjmatrix)
        boolfunc = self.boolfunc
        
        # computing the derivatives
        p = self.p
        ydot[0] = p.alpha1[0, 0] * R_logic(
            Hill(Y[reglist[0, 0]], p.theta11[0, 0], p.p11[0, 0]), 
            Hill(Y[reglist[0, 1]], p.theta12[0, 0], p.p12[0, 0]), 
            boolfunc[0]) - p.gamma1[0, 0] * y[0]
        ydot[1] = p.alpha1[0, 1] * R_logic(
            Hill(Y[reglist[0, 0]], p.theta11[0, 1], p.p11[0, 1]), 
            Hill(Y[reglist[0, 1]], p.theta12[0, 1], p.p12[0, 1]), 
            boolfunc[0]) - p.gamma1[0, 1] * y[1]
        ydot[2] = p.alpha2[0, 0] * R_logic(
            Hill(Y[reglist[1, 0]], p.theta21[0, 0], p.p21[0, 0]), 
            Hill(Y[reglist[1, 1]], p.theta22[0, 0], p.p22[0, 0]), 
            boolfunc[1]) - p.gamma2[0, 0] * y[2]
        ydot[3] = p.alpha2[0, 1] * R_logic(
            Hill(Y[reglist[1, 0]], p.theta21[0, 1], p.p21[0, 1]), 
            Hill(Y[reglist[1, 1]], p.theta22[0, 1], p.p22[0, 1]), 
            boolfunc[1]) - p.gamma2[0, 1] * y[3]
        ydot[4] = p.alpha3[0, 0] * R_logic(
            Hill(Y[reglist[2, 0]], p.theta31[0, 0], p.p31[0, 0]), 
            Hill(Y[reglist[2, 1]], p.theta32[0, 0], p.p32[0, 0]), 
            boolfunc[2]) - p.gamma3[0, 0] * y[4]
        ydot[5] = p.alpha3[0, 1] * R_logic(
            Hill(Y[reglist[2, 0]], p.theta31[0, 1], p.p31[0, 1]), 
            Hill(Y[reglist[2, 1]], p.theta32[0, 1], p.p32[0, 1]), 
            boolfunc[2]) - p.gamma3[0, 1] * y[5]

    def check_adjacency_and_boolfunc(self, adjmatrix, boolfunc):
        """
        Check for consistency between *adjmatrix* and *boolfunc*.
        
        The number of non-zero elements in the i-th row of *adjmatrix* and the
        i-th element of *boolfunc*, both indicate the number of regulators (0, 
        1 or 2) of gene i-1. An exception is raised if the numbers don't match.
        
        
        >>> s = Sigmoidmodel()
        >>> adjmatrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        Exception raised when *adjmatrix* indicates 0 regulators for gene 1 
        and *boolfunc* indicates 2      
        
        >>> s.check_adjacency_and_boolfunc(adjmatrix, [1, 5, 7])   
        Traceback (most recent call last):
        ...
        Exception: Mismatch: adjmatrix[0, :] indicates 0 regulator(s), 
        but boolfunc[0] does not.
        
        Check passes when *adjmatrix* and *boolfunc* both indicate 0 regulators
        for gene 1
        
        >>> s.check_adjacency_and_boolfunc(adjmatrix, [15, 5, 5])
        """
        boolfunc = np.array(boolfunc)
        
        if not adjmatrix.shape == (3, 3):
            raise Exception('adjmatrix.shape is not (3, 3)')
        if not boolfunc.shape == (3,):
            raise Exception('boolfunc does not have 3 elements')
        
        
        msg = "Mismatch: adjmatrix[{}, :] indicates {} regulator(s), "
        msg += "but boolfunc[{}] does not."
        for i in range(3):
            if np.shape(adjmatrix[i, :].nonzero()[0]) == (0,):
                # adjmatrix indicates no regulators
                if ~np.max(boolfunc[i] == [15, 16]):
                    raise Exception(msg.format(i, 0, i))
            elif np.shape(adjmatrix[i, :].nonzero()[0]) == (1,):
                # adjmatrix indicates one regulator
                if ~np.max(boolfunc[i] == [5, 7]):
                    raise Exception(msg.format(i, 1, i))
            elif np.shape(adjmatrix[i, :].nonzero()[0]) == (2,):
                # adjmatrix inicates three regulators
                if ~np.max(boolfunc[i] == [1, 2, 3, 4, 9, 10, 11, 12, 13, 14]):
                    raise Exception(msg.format(i, 2, i))
            elif np.shape(adjmatrix[i, :].nonzero()[0]) == (3,):
                raise Exception('Only 0, 1 or 2 regulators allowed per gene')


    def adjacency_to_reglist(self, adjmatrix):
        """
        Convert adjacency matrix to nested lists of regulators  
        
        :return reglist: 3-element list of indexes of regulator gene for each gene such 
            that gene i is a regulator of gene j iff reglist[j] contains i (for i,j=0,1,2)
            a value of 3 means no regulator (see Y[3] in equations_diploid_adjacency)
        :return boolfunc: 3-element list of integers indicating the Boolean function 
            used in the gene regulation functions.
        
        >>> s = Sigmoidmodel()
        >>> adjmatrix = np.array([[0, 0, 0], [1, 0, 0], [0, -1, 0]])
        >>> s.adjacency_to_reglist(adjmatrix)
        array([[3, 3], [0, 3], [1, 3]])
        >>> adjmatrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        >>> s.adjacency_to_reglist(adjmatrix)
        Traceback (most recent call last):
            ...
        Exception: Only 0,1 or 2 regulators allowed per gene

        """
        reglist = np.zeros((3, 2), dtype=int)
                
        for i in range(3):
            if np.shape(adjmatrix[i, :].nonzero()[0]) == (0,):
                # No regulators
                reglist[i, :] = [3, 3]
            elif np.shape(adjmatrix[i, :].nonzero()[0]) == (1,):
                # One regulator
                reglist[i, :] = [adjmatrix[i, :].nonzero()[0][0], 3]
            elif np.shape(adjmatrix[i, :].nonzero()[0]) == (2,):
                # Two regulators
                reglist[i, :] = adjmatrix[i, :].nonzero()[0]
            else: 
                raise Exception('Only 0,1 or 2 regulators allowed per gene')
        
        return reglist
      
    def adjacency_to_boolfunc(self, adjmatrix):
        """
        Convert adjacency matrix to a list of boolean functions 
        (index used as input to :meth:`~cgp.sigmoidmodels.doseresponse.R_logic`). The number of 
        nonzero elements in the i-th row 
        of adjmatrix indicates the number of regulators (0,1,2) of gene [i]. This function 
        creates a valid list of boolean functions directly from adjmatrix by restricting the 
        number of functions. 
        
        * In the case of no regulators the index is 16 (always on). 
        * In the case of one regulator the sign of the nonzero element determines, 
          a 1 in adjmatrix means an activation an return index 5 (X1), a -1 in adjmatrix means
          repression an return index 7 (NOT(X1))
        * With two regulators the sign of the nonzero elements determines the mode of regulation
          (activation/repression) as for one regulator, and an AND function (indexes 1-4) 
          is used to combine the two regulators.
          
        See also function R_logic. 
        
        :return boolfunc:  3-element list of integers indicating the Boolean function 
            used in the gene regulation functions.
            
        >>> s = Sigmoidmodel()
        >>> adjmatrix = np.array([[0, 0, 0], [1, 0, 0], [0, -1, 0]])
        >>> s.adjacency_to_boolfunc(adjmatrix)
        array([ 16., 5., 7.])
        """
        
        boolfunc = np.nan * np.zeros(3,)
        
        for i in range(3):
            if np.shape(adjmatrix[i, :].nonzero()[0]) == (0,):
                # No regulators
                boolfunc[i] = 16  # always on
            elif np.shape(adjmatrix[i, :].nonzero()[0]) == (1,):
                # One regulator
                if adjmatrix[i, adjmatrix[i, :].nonzero()[0][0]] == 1:
                    boolfunc[i] = 5
                else:
                    boolfunc[i] = 7
            elif np.shape(adjmatrix[i, :].nonzero()[0]) == (2,):
                # Two regulators
                # Extract regulatory mode (1: activation, -1: inhibition)
                regmode = adjmatrix[i, adjmatrix[i, :].nonzero()[0]]
                if np.min(regmode == np.array([1, 1])):
                    boolfunc[i] = 1
                elif np.min(regmode == np.array([1, -1])):
                    boolfunc[i] = 2
                elif np.min(regmode == np.array([-1, 1])):
                    boolfunc[i] = 3
                elif np.min(regmode == np.array([-1, -1])):
                    boolfunc[i] = 4
                else:
                    raise Exception('Only -1 and 1 allowed in adjacency matrix')
            else: 
                raise Exception('Only 0,1 or 2 regulators allowed per gene')
        
        return boolfunc      

    def sample_hetpar(self):
        """
        Sample allelic parameter values uniformly.
        
        Mean = self.par_mean, span = self.par_span
        
        :return hetpar: numpy recarray like self.p with parameter values for 
            an all-loci heterozygote.
        
        >>> np.random.seed(0)
        >>> s = Sigmoidmodel()
        >>> s.sample_hetpar() # doctest: +ELLIPSIS
        rec.array([ ([154.8813503927325, 171.51893663724195], ... [5.105352989948937, 6.115905539817836], [10.0, 10.0])], 
        dtype=[('alpha1', '<f8', (2,)), ... ('p32', '<f8', (2,)), ('gamma3', '<f8', (2,))])
        """
        low = self.par_mean - self.par_span/2
        high = self.par_mean + self.par_span/2
        hetpar = np.row_stack([np.random.uniform(size=(1, 2), low=l, high=h) 
            for l, h in zip(low, high)])
        hetpar = hetpar.flatten().view(self.p.dtype, np.recarray)
        return hetpar

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)