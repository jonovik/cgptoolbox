"""
Genotype-to-parameter maps

"""
import numpy as np

def flatten(nestedList):
    """
    Flatten a nested list structure.
    
    Source:
    http://wiki.python.org/moin/ProblemSets/99%20Prolog%20Problems%20Solutions#Problem7.3AFlattenanestedliststructure
    """
    def aux(listOrItem):
        if isinstance(listOrItem, list):
            for elem in listOrItem:
                for item in aux(elem):
                    yield item
        else:
            yield listOrItem
    
    return list(aux(nestedList))

def monogenicpar(genotype, hetpar, relvar=0.5, nloci=[], absvar=None):
    """
    Genotype-to-parameter map that assumes one biallelic locus per parameter
    
    Returns a record array of parameter values for the given genotype.
    Input:
        nloci: geno2par argument not used here
        genotype : sequence where each item is 0, 1, or 2, 
            denoting the "low" homozygote, the "baseline" heterozygote, and 
            the "high" homozygote, respectively. Alternatively, each item can 
            be a 2-tuple where each item is 0 or 1.
        hetpar : record array of "fully heterozygous" parameter values
        relvar : proportion change (scalar)
        absvar : absolute change (array-like), overrides relvar if present
    Gene/parameter names are taken from the fieldnames of hetpar.
    Thus, the result has the same dtype (Numpy data type) as the genotype array.
    
    Creating a record array of baseline parameter values 
    (corresponding to the fully heterozygous genotype).
    
    >>> baseline = [("a", 0.25), ("b", 0.5), ("theta", 0.75), ("I", 1.0)]
    >>> dtype = [(k, float) for k, v in baseline]
    >>> hetpar = np.array([v for k, v in baseline]).view(dtype, np.recarray)
    
    Parameter values for four "loci" with 
    (low homozygote, heterozygote, heterozygote, high homozygote).
    
    >>> genotype = [[0, 0], [0, 1], [1, 0], [1, 1]]
    >>> monogenicpar(genotype, hetpar)
    rec.array([(0.125, 0.5, 0.75, 1.5)], 
          dtype=[('a', '<f8'), ('b', '<f8'), ('theta', '<f8'), ('I', '<f8')])
    
    Specifying relative or absolute parameter variation.
    
    >>> monogenicpar(genotype, hetpar, relvar = 1.0)
    rec.array([(0.0, 0.5, 0.75, 2.0)],...
    >>> monogenicpar(genotype, hetpar, absvar = [0.125, 0.25, 0.5, 0.75])
    rec.array([(0.125, 0.5, 0.75, 1.75)],...
    """
    genotype = np.array(genotype)
    if genotype.ndim == 2:
        allelecount = np.array(genotype).sum(axis=1)
    else:
        allelecount = genotype
    result = hetpar.copy().view(float)
    if absvar is None:
        absvar = result * relvar
    else:
        absvar = np.array(absvar).view(float)
    result += (allelecount - 1) * absvar
    return result.view(hetpar.dtype, np.recarray)

def geno2par_additive(genotype, hetpar, relvar=0.5, nloci=0, absvar=None):
    """
    General (many:many) additive genotype-to-parameter map for N biallelic (alleles 0/1) 
    loci affecting M parameters. The gp-map is intra- and inter-locus additive and for all loci
    genotype (0,0) (coded 0)  has the lowest parameter value.
    
    Returns a record array of parameter values for the given genotype.
    Input:
        genotype : a single, multilocus genotype represented by a sequence where each item is 0, 1, or 2, 
            denoting the "low" homozygote, the "baseline" heterozygote, and 
            the "high" homozygote, respectively. Alternatively, each item can 
            be a 2-tuple where each item is 0 or 1.
        hetpar : record array of  M parameter values for the full heterozygot genotype [[0,1],[0,1],..,[0,1],[0,1]]
        nloci:	 (int) number of polymorphic loci
        relvar : proportional change in parameter values associated with changing the genotype at a locus from
            heterozygot to either of the homozygots
            (scalar float) - 1 locus: 1 parameter , same proportion for all loci
            (NxM array)- element i,j gives the proportional change in parameter j associated with genotype change at locus i.
            (list with N dictionaries) - sparse version of the NxM array, dictionary i contains non-zero parameternammes as 
                keys and proportional changes as values.
        absvar : absolute change (array-like), overrides relvar if present

    Gene/parameter names are taken from the fieldnames of hetpar.
    Thus, the result has the same dtype (Numpy data type) as the genotype array.
    
    Creating a record array of baseline parameter values 
    (corresponding to the fully heterozygous genotype).
    
    >>> baseline = [("a", 0.25), ("b", 0.5), ("theta", 0.75)]
    >>> dtype = [(k, float) for k, v in baseline]
    >>> hetpar = np.array([v for k, v in baseline]).view(dtype, np.recarray)
    
    Introducing two loci A and B such that A affects parameters a and b, while B affects
    parameters b and theta
    
    >>> loci = ["Locus A","Locus B"]
    >>> relvar = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5]])

    Parameter values for genotype aaBB
    
    >>> genotype = [[0, 0], [1, 1]]
    >>> geno2par_additive(genotype, hetpar, relvar, 2)
    rec.array([(0.125, 0.5, 1.125)], 
          dtype=[('a', '<f8'), ('b', '<f8'), ('theta', '<f8')])
    
    List of dicts or float for relvar
    
    >>> relvar = [{'a':0.5,'b':0.5},{'b':0.5,'theta':0.5}] #same relvar as above
    >>> geno2par_additive(genotype, hetpar, relvar, 2)
    rec.array([(0.125, 0.5, 1.125)], 
          dtype=[('a', '<f8'), ('b', '<f8'), ('theta', '<f8')])
    >>> genotype = [0, 1, 2]
    >>> geno2par_additive(genotype, hetpar, relvar = 1.0, nloci=3)
    rec.array([(0.0, 0.5, 1.5)],...
    """

    N = nloci
    M = len(hetpar[0])

    ##convert genotype to (1,N) np.array with elements in 0,1,2
    genotype = np.array(genotype)
    if genotype.ndim == 2:
        allelecount = np.array(genotype).sum(axis=1)
    else:
        allelecount = genotype
    allelecount.shape=(1,N) 

    ## Check input type of relvar and convert to (N,M) np.array
    # TODO (kapsle, kapsle): Raise errors instead of prints
    if type(relvar).__name__=='ndarray':
	if relvar.shape!=(N, M):
	    print 'relvar.shape must be (nloci,M)'
    elif type(relvar).__name__=='float': #Test for N==M and create diagonal array
	if N==M:
	    relvar = np.eye(N) * relvar
	else:
	    print 'nloci!=M so scalar relvar is not accepted' 
    elif type(relvar).__name__=='list': #Test length, create NxM array
	if len(relvar)==N:
	    rvar = np.zeros((N,M)).view(hetpar.dtype, np.recarray)  #initialize with zeros
	    for i in range(N): #fill in non-zeros row by row
		[ setattr(rvar[i],key,value) for key,value in relvar[i].items() ]
	    relvar = rvar.view(float)
	else:
	    print 'relvar list must contain nloci dictionaries'
    else:
	print 'Relvar must be float, numpy array or list' 
   
    ## calculate parameter values for genotype
    basis = hetpar.copy().view(float)
    if absvar is None:
        absvar = basis * relvar
    else:
        absvar = np.array(absvar).view(float)
    basis += np.reshape(np.dot(allelecount-1, absvar),(M,))
    return basis.view(hetpar.dtype, np.recarray)
 
def prepare_geno2par_additive(genes, parnames, origpar, relchange, nloci=[]):
    '''
    Prepare input for geno2par_additive, works for 1:1 and 1:many
    
    Input:
        genes       list of gene names in cGP model
        parnames    nested list of heritable parameters associated with each gene in genes
        origpar     list of numpy arrays with original parameter values for all parameters in parnames
        relchange   nested list of numpy array with relative change (from origpar) for 00 homozygote and 11 homozygote respectively
        nloci       number of polymorloci (defaults to len(genes))
    
    Output:
        loci        list of polymorphic genes
        hetpar      recarray with parameter values for complete heterozygot
        relvar      see input to geno2par_additive
    
    TODO: Implement many:many 
    
    Example:
    
    >>> np.random.seed(1)
    >>> genes = ['GeneA','GeneB']
    >>> parnames = [['ParA'],['ParB','ParC']]
    >>> origpar = [np.array([1]), np.array([2, 3])]
    >>> relchange = [np.array([[0.5, 1.5]]), np.array([[0.8, 1.1],[0.8, 1.2]])]
    >>> loci, hetpar, relvar = prepare_geno2par_additive(genes, parnames, origpar, relchange)
    >>> loci
    ['GeneA', 'GeneB']
    >>> hetpar                                      #doctest: +NORMALIZE_WHITESPACE
    rec.array([(1.0, 1.9000000000000001, 3.0)],
       dtype=[('ParA', '<f8'), ('ParB', '<f8'), ('ParC', '<f8')])
    >>> relvar                                      #doctest: +NORMALIZE_WHITESPACE
    array([[ 0.5       ,  0.        ,  0.        ],
           [ 0.        ,  0.15789474,  0.2       ]])
    
    '''
    
    if nloci==[]:
        nloci = len(genes)

    #sample polymorphic loci
    loci = range(len(genes))
    np.random.shuffle(loci)
    loci = loci[0:nloci]
    genes = [ genes[i] for i in loci ]
    parnames = [ parnames[i] for i in loci ]
    relchange = [ relchange[i] for i in loci ]
    origpar = np.concatenate(tuple([origpar[i] for i in loci]))

    #format names and change to recarrays hetpar and relvar
    dtype = [(par,float) for par in flatten(parnames)]
    
    hetrel = np.concatenate(tuple([np.mean(ch,axis=1) for ch in flatten(relchange)]))
    hetpar = origpar * hetrel
    hetpar = hetpar.view(dtype,np.recarray)
    relvar = np.zeros((len(genes),len(flatten(parnames))))
    par = 0
    for i in range(len(genes)):
        for j in range(len(parnames[i])):               
            relvar[i,par] = 1
            par += 1 
    relvar = relvar * np.concatenate(tuple([(np.mean(ch,axis=1)-np.min(ch,axis=1))/np.mean(ch,axis=1) for ch in flatten(relchange)]))

    return genes, hetpar, relvar

def geno2par_diploid(genotype, hetpar, relvar):
    """
    General (many:many) diploid genotype-to-parameter map for N biallelic (alleles 0/1) 
    loci affecting M parameters. With a diploid parameter genotype-to-parameter map we mean
    that each allele of a genotype is mapped to one of two parameters in a tuple. These parameter-pairs 
    are homologous in the sence that they describe functional aspects of two homologous DNA sequences.
    
    Returns a record array of tuples with shape (2,) containing parameter values for the given genotype.

    Input:
        genotype : a single, multilocus genotype represented by a sequence where each item is 0, 1, or 2, 
            denoting the "low" homozygote, the "baseline" heterozygote, and 
            the "high" homozygote, respectively. Alternatively, each item can 
            be a 2-tuple where each item is 0 or 1.
        hetpar : record array of  M (2,) tuples of parameter values for the full heterozygot genotype [[0,1],[0,1],..,[0,1],[0,1]]
        relvar : proportional change in parameter values associated with changing the genotype at a locus from
            heterozygot to either of the homozygots
            (scalar float) - 1 locus: 1 parameter , same proportion for all loci
            (NxM array)- element i,j gives the proportional change in parameter j associated with genotype change at locus i.
            (list with N dictionaries) - sparse version of the NxM array, dictionary i contains non-zero parameternammes as 
                keys and proportional changes as values.
    

    Gene/parameter names are taken from the fieldnames of hetpar.
    Thus, the result has the same dtype (Numpy data type) as the genotype array.
    
    Creating a record array of baseline parameter values 
    (corresponding to the fully heterozygous genotype).
    
    >>> baseline = [("a", (0.25,0.5)), ("b", (0.5,1.5)), ("c", (0.75,1.25))]
    >>> dtype = [(k, float, (2,)) for k, v in baseline]
    >>> hetpar = np.array([v for k, v in baseline]).flatten().view(dtype, np.recarray)
    
    Introducing two loci A and B such that A affects parameters a and b, while B affects
    parameters c
    
    >>> loci = ["Locus A","Locus B"]
    >>> relvar = np.array([[1, 1, 0], [0, 0, 1]])

    Parameter values for genotype aaBB
    
    >>> genotype = [[0, 0], [1, 1]]
    >>> geno2par_diploid(genotype, hetpar, relvar)      #doctest: +NORMALIZE_WHITESPACE
    rec.array([([0.25, 0.25], [0.5, 0.5], [1.25, 1.25])],
          dtype=[('a', '<f8', (2,)), ('b', '<f8', (2,)), ('c', '<f8', (2,))])
 

    """
    from copy import copy

    N = relvar.shape[0] #number of genes
    M = relvar.shape[1] #number of heritable parameters

    ##convert genotype to (1,N) np.array with elements in 0,1,2
    genotype = np.array(genotype)
    if genotype.ndim == 2:
        allelecount = np.array(genotype).sum(axis=1)
    else:
        allelecount = genotype
    allelecount.shape=(N,) 

    genopar = copy(hetpar)    #copy parameter structure
    #modify parameter values for each single locus genotype
    for i in range(N):
        for j in range(M):
            if relvar[i,j]==1:
                if allelecount[i]==0:
                    genopar[0][j][1] = hetpar[0][j][0]
                if allelecount[i]==2:
                    genopar[0][j][0] = hetpar[0][j][1]
                
    return genopar
    

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
