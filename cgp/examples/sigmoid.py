"""
Basic example of cGP study using sigmoid model of gene regulatory networks.
"""
# pylint: disable=W0621

import numpy as np
from matplotlib import pyplot as plt

from cgp.utils.placevalue import Placevalue	
from cgp.gt.gt2par import geno2par_diploid
from cgp.sigmoidmodels.sigmoid_cvode import Sigmoidmodel
from cgp.phenotyping.attractor import AttractorMixin
from cgp.utils.recfun import cbind, restruct

# ode model  
class Model(Sigmoidmodel,AttractorMixin):
  pass

model = Model()


# Allelic parameter values sampled as defined in Gjuvsland et al. (2007)
hetpar = model.sample_hetpar()
relvar = np.zeros((3,18))
relvar[0,0:6] = 1
relvar[1,6:12] = 1
relvar[2,12:18] = 1
names = ['Gene1','Gene2','Gene3']

# Enumerate genotypes
genotypes = Placevalue([3,3,3],names=names,msd_first=False)


def gt2par(gt,hetpar,relvar):
    """Genotype-to-parameter mapping."""
    return geno2par_diploid(gt,hetpar,relvar)

def par2ph(par):
    """Parameter-to-phenotype mapping."""
    model._ReInit_if_required(y=np.array([0.0,0.0,0.0,0.0,0.0,0.0]))
    model.pr[:] = par
    t, y = model.eq(tmax=100,tol=1e-6)
    y = np.array([y[0]+y[1],y[2]+y[3],y[4]+y[5]])
    y.dtype=[("Y1", float), ("Y2", float), ("Y3", float)]
    return y.view(np.recarray)

def cgpstudy():
    """
    Basic example of connecting the building blocks of a cGP study.
    
    This implements the simulation pipeline used in Gjuvsland et al. (2007) and
    Gjuvsland et al. (2011)
    
    :TODO: save and read into R
    :TODO: in R create classic lineplots for 3-locus GP mapping
    :TODO: in R use noia package to partition variance
    """
    from numpy import concatenate as cat

    gt = np.array(genotypes)
    par = cat([gt2par(list(g), hetpar, absvar) for g in gt])
    ph = cat([par2ph(p) for p in par])
     



if __name__ == "__main__":
    cgpstudy()
