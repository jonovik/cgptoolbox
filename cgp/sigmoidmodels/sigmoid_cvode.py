"""
Module implementing three-gene regulatory network models using the sigmoid
formalism (Plahte et al. 1998), in the diploid form introduced by Omholt et al.
(2000) with one differential equation per allele. The most class Sigmoidmodel
inherits from Cellmlmodel and contains an adjacency matrix defining the
connectivity of the network, a function defining the right hand side of the ODEs
and a function for sampling heritabel variation in allelic parameter values for
cGP studies. The parameter ranges default to the ranges used in Gjuvsland et al.
(2007).

References:

* Plahte E, Mestl T, Omholt SW (1998)
  `A methodological basis for description and analysis of systems with complex
  switch-like interactions <http://dx.doi.org/10.1007/s002850050103>`_ J. Math.
  Biol. 36: 321-348.

* Omholt SW, Plahte E, Oyehaug L and Xiang K (2000)
  `Gene regulatory networks generating the phenomena of additivity, dominance
  and epistasis <http://www.genetics.org/content/155/2/969.full>`_ Genetics
  155(2): 969-980.
  
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
    Class to solve sigmoid ode model equations.
    
    FIXME: Inherits from Namedcvodeint, but the default y is not a recarray.
    """
    
    #: Adjacency matrix defining network connectivity
    adjmotif = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    
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

    def __init__(self, t=(0, 1), y=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 
        p=p, adjmotif=adjmotif):
        """Constructor."""
        t = np.array(t)
        y = np.asanyarray(y)
        self.adjmotif = adjmotif
        super(Sigmoidmodel, self).__init__(self.equations_diploid_adjacency, 
            t, y, p=p)
    
    def equations_diploid_adjacency(self, t, y, ydot, fdata):
        """
        ODE rate function for a diploid model with three loci X1, X2, X3.
        
        This is a general model where each gene can be regulated by two of the 
        three genes. 
        The actual motif is controlled by the adjacency matrix adjmotif.
        
        :TODO: for JO use adjmotifs as input to networkX to create plot of 
            directed graph 
        """
        
        Y = np.nan*np.ones((4,))
        Y[0] = y[0] +y[1]    # sum of x11 and x12
        Y[1] = y[2] + y[3]   # sum of x21 and x22
        Y[2] = y[4] + y[5]   # sum of x31 and x32
        Y[3] = np.nan        # dummy variable used when there is no regulator
        
        regulators, grfmotifs = self.adjacency_to_grf()
        
        # computing the derivatives
        p = self.p
        ydot[0] = p.alpha1[0, 0] * R_logic(
            Hill(Y[regulators[0, 0]], p.theta11[0, 0], p.p11[0, 0]), 
            Hill(Y[regulators[0, 1]], p.theta12[0, 0], p.p12[0, 0]), 
            grfmotifs[0]) - p.gamma1[0, 0] * y[0]
        ydot[1] = p.alpha1[0, 1] * R_logic(
            Hill(Y[regulators[0, 0]], p.theta11[0, 1], p.p11[0, 1]), 
            Hill(Y[regulators[0, 1]], p.theta12[0, 1], p.p12[0, 1]), 
            grfmotifs[0]) - p.gamma1[0, 1] * y[1]
        ydot[2] = p.alpha2[0, 0] * R_logic(
            Hill(Y[regulators[1, 0]], p.theta21[0, 0], p.p21[0, 0]), 
            Hill(Y[regulators[1, 1]], p.theta22[0, 0], p.p22[0, 0]), 
            grfmotifs[1]) - p.gamma2[0, 0] * y[2]
        ydot[3] = p.alpha2[0, 1] * R_logic(
            Hill(Y[regulators[1, 0]], p.theta21[0, 1], p.p21[0, 1]), 
            Hill(Y[regulators[1, 1]], p.theta22[0, 1], p.p22[0, 1]), 
            grfmotifs[1]) - p.gamma2[0, 1] * y[3]
        ydot[4] = p.alpha3[0, 0] * R_logic(
            Hill(Y[regulators[2, 0]], p.theta31[0, 0], p.p31[0, 0]), 
            Hill(Y[regulators[2, 1]], p.theta32[0, 0], p.p32[0, 0]), 
            grfmotifs[2]) - p.gamma3[0, 0] * y[4]
        ydot[5] = p.alpha3[0, 1] * R_logic(
            Hill(Y[regulators[2, 0]], p.theta31[0, 1], p.p31[0, 1]), 
            Hill(Y[regulators[2, 1]], p.theta32[0, 1], p.p32[0, 1]), 
            grfmotifs[2]) - p.gamma3[0, 1] * y[5]

    def adjacency_to_grf(self):
        """
        Convert adjacency matrix of Sigmoidmodel instance to nested lists of 
        regulators and gene regulation functions.
        
        :return regulators: 3-element list of indexes of regulator gene for each gene such 
            that gene i is a regulator of gene j iff regulators[j] contains i
        :return grfmotifs:  3-element list of integers indicating the Boolean function 
            used in the gene regulation functions.
        """
        adjmotif = self.adjmotif
        regulators = np.nan * np.zeros((3, 2))
        grfmotifs = np.nan * np.zeros(3,)
        
        for i in range(3):
            if np.shape(adjmotif[i, :].nonzero()[0]) == (0,):
                # No regulators
                regulators[i, :] = [3, 3]
                grfmotifs[i] = 16  # always on
            elif np.shape(adjmotif[i, :].nonzero()[0]) == (1,):
                # One regulator
                regulators[i, :] = [adjmotif[i, :].nonzero()[0][0], 3] 
                if adjmotif[i, regulators[i, 0]] == 1:
                    grfmotifs[i] = 5
                else:
                    grfmotifs[i] = 7
            elif np.shape(adjmotif[i, :].nonzero()[0]) == (2,):
                # Two regulators
                regulators[i, :] = adjmotif[i, :].nonzero()[0]
                # TODO: Figure out the meaning of the magic number below
                magic = adjmotif[i, adjmotif[i, :].nonzero()[0]]
                if np.min(magic == np.array([1, 1])):
                    grfmotifs[i] = 1
                elif np.min(magic == np.array([1, -1])):
                    grfmotifs[i] = 2
                elif np.min(magic == np.array([-1, 1])):
                    grfmotifs[i] = 3
                elif np.min(magic == np.array([-1, -1])):
                    grfmotifs[i] = 4
                else:
                    raise Exception('Only -1 and 1 allowed in adjacency matrix')
            else: 
                raise Exception('Only 0,1 or 2 regulators allowed per gene')
        
        return regulators, grfmotifs

    def sample_hetpar(self):
        """
        Sample allelic parameter values uniformly.
        
        Mean = self.par_mean, span = self.par_span
        
        :return hetpar: numpy recarray like self.p with parameter values for 
            an all-loci heterozygote.
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
