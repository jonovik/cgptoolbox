"""
Enumerate haploid or diploid genotypes with two alleles at *n* loci.

.. inheritance-diagram:: cgp.utils.placevalue.Placevalue Genotype

These classes are suitable for studies enumerating or sampling from all possible
genotypes. For a single locus, the three genotypes *aa, Aa, AA* map to 0, 1, 2
respectively. Thus, for instance, the three-locus genotype *AabbCC* would be
``[1, 0, 2]``. Such a vector representation of a genotype could be fed into a
genotype-to-parameter map function. An efficient mapping of each genotype to a
unique integer index facilitates easy sampling and indexing. We can list all
vectors/genotypes of a placevalue/genotype object by converting it to an array.

Here is an example with two biallelic loci::

    >>> pv = Placevalue([3, 3])
    >>> pv.int2vec(0)  # Genotype aabb
    array([0, 0])
    >>> AaBB = [1, 2]
    >>> pv.vec2int(AaBB)  # 1 * 3**1 + 2 * 3**0
    5
    >>> np.array(pv)
    array([[0, 0],
           [0, 1],
           [0, 2],
           [1, 0],
           [1, 1],
           [1, 2],
           [2, 0],
           [2, 1],
           [2, 2]])

The difference between class Genotype and Placevalue lies in how they order
genotypes. Class :class:`Genotype` puts  heterozygotes first, ensuring that 
the first :math:`3^k` genotypes make up a :math:`3^k` full factorial design 
in the first *k* parameters::

    >>> np.array(Genotype([3, 3]))
    array([[1, 1],
           [1, 0],
           [1, 2],
           [0, 1],
           [0, 0],
           [0, 2],
           [2, 1],
           [2, 0],
           [2, 2]])
"""

import numpy as np

from ..utils.placevalue import Placevalue
from ..utils.unstruct import unstruct

class Genotype(Placevalue):
    """
    Enumerate haploid or diploid genotypes with two alleles at *n* loci.
    
    Within each locus, levels are given in order [1, 0, 2], for 
    the baseline heterozygote, low homozygote, and high homozygote.
    This ensures that the first 3**k genotypes make up a full factorial design 
    with k factors, keeping the remaining n-k loci as heterozygotes.
    
    See :mod:`genotype` for examples.
    
    If a *names* argument is given, *n* can be omitted and defaults to 
    [3] * len(names).
    
    >>> Genotype(names=["a", "b"])
    Genotype(rec.array([(3, 3)], dtype=[('a', '...'), ('b', '...

    """
    code = np.array([1, 0, 2])
    def int2vec(self, i):
        """
        Vector [i0, i1, ...] corresponding to an integer. Code order = 1, 0, 2.
        
        >>> gt = Genotype(np.rec.fromrecords([[3, 3]], names=["a", "b"]))

        The heterozygote (baseline) scenario comes first.
        
        >>> gt.int2vec(0) == np.array([(1, 1)], dtype=[('a', int), ('b', int)])
        array([ True], dtype=bool)        
        >>> np.concatenate(gt)
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
        >>> v = np.concatenate(gt)
        >>> gt.vec2int(v)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=object)
        """
        return super(Genotype, self).vec2int(self.code[unstruct(v)])
    
    def __init__(self, n=None, msd_first=True, names=None):
        """Constructor for class :class:`Genotype`."""
        if not n:
            n = [3] * len(names)
        msg = "Genotype only implemented for biallelic loci"
        assert all(unstruct(n).squeeze() <= 3), msg
        super(Genotype, self).__init__(n, msd_first, names)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
