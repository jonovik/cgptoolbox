"""Genotype: Full factorial design for three levels of a variable."""

import numpy as np

from ..utils.placevalue import Placevalue
from ..utils.unstruct import unstruct

class Genotype(Placevalue):
    """
    Genotype: Full factorial design for three levels of a variable.
    
    Within each locus, levels are given in order [1, 0, 2], for 
    the baseline heterozygote, low homozygote, and high homozygote.
    This ensures that the first 3**k genotypes make up a full factorial design 
    with k factors, keeping the remaining n-k loci as heterozygotes.
    """
    code = np.array([1, 0, 2])
    def int2vec(self, i):
        """
        Vector [i0, i1, ...] corresponding to an integer. Code order = 1, 0, 2.
        
        >>> gt = Genotype(np.rec.fromrecords([[3, 3]], names=["a", "b"]))

        The heterozygote (baseline) scenario comes first.
        
        >>> gt.int2vec(0) == np.array([(1, 1)], dtype=[('a', int), ('b', int)])
        array([ True], dtype=bool)        
        >>> np.concatenate(list(gt))
        array([(1, 1), (1, 0), (1, 2), 
               (0, 1), (0, 0), (0, 2), 
               (2, 1), (2, 0), (2, 2)], dtype=[('a', '<i...'), ('b', '<i...')])
        """
        v = super(Genotype, self).int2vec(i)
        vi = v.view(self.fieldtype)
        vi[:] = self.code[vi]
        return v
    
    def vec2int(self, v):
        """
        Integer corresponding to a vector [i0, i1, ...]. Code order = 1, 0, 2.
        
        >>> gt = Genotype(np.rec.fromrecords([[3, 3]], names=["a", "b"]))
        >>> gt.vec2int([1, 1])
        0
        >>> v = np.concatenate(list(gt))
        >>> gt.vec2int(v)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        """
        return super(Genotype, self).vec2int(self.code[unstruct(v)])
    
    def __init__(self, n, msd_first=True):
        """
        >>> np.array(Genotype([3, 2]))
        array([[1, 1],
               [1, 0],
               [0, 1],
               [0, 0],
               [2, 1],
               [2, 0]])
        >>> Genotype([4, 4])
        Traceback (most recent call last):
        AssertionError: Genotype only implemented for biallelic loci
        """
        msg = "Genotype only implemented for biallelic loci"
        assert all(unstruct(n).squeeze() <= 3), msg
        super(Genotype, self).__init__(n, msd_first)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
