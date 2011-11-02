"""
Mapping of integers to vectors [i0, i1, ...] with [n0, n1, ...] possible values.

A Placevalue object can be indexed, iterated over, or converted to a list, 
returning vectors corresponding to successive integers.
Conversely, vec2int() gives the integer corresponding to a given vector.

USAGE

Default is most significant digit first, so that the last digit changes fastest.

>>> p = Placevalue([2, 3, 4])

Iteration over vectors.

>>> for i in p: print i                                     # doctest: +ELLIPSIS
[0 0 0]
[0 0 1]
...
[1 2 2]
[1 2 3]

Indexing.

>>> p[13]                   # 1 * (3*4) + 0 * (4) + 1
array([1, 0, 1])

Conversion from vector to integer.

>>> p.vec2int([1,2,3])      # 1 * (3*4) + 2 * (4) + 3
23

Converting to list, equivalent to [i for i in p].

>>> list(p)                                                 # doctest: +ELLIPSIS
[array([0, 0, 0]), array([0, 0, 1]), ..., array([1, 2, 3])]

A single array is more compact.

>>> np.array(list(p))                                       # doctest: +ELLIPSIS
array([[0, 0, 0], [0, 0, 1], ..., [1, 2, 3]])

NAMED POSITIONS

Using a structured array with named fields to construct the Placevalue object.

>>> dtype = [("a", np.int16), ("b", np.int16)]
>>> pr = Placevalue(np.rec.fromrecords([[2,3]], dtype=dtype))
>>> pr[7]
array([(2, 1)], dtype=[('a', '<i2'), ('b', '<i2')])
>>> np.concatenate(list(pr))                                # doctest: +ELLIPSIS
array([(0, 0), (0, 1), ..., (1, 1), (1, 2)], dtype=[('a', '<i2'), ('b', '<i2')])

DETAILS

The i-vectors correspond to a place-value system with prod(n) unique values.
If i[0] is the least significant digit, position j has value prod(n[:j]).
The default is to have i[0] as the most significant digit, as when reading 
numbers left-to-right.

The integer is computed by multiplying each vector element with its place value 
and summing the result, just like 123 = 1*100 + 2*10 + 3*1.

This module also has a Python 2.5.2 replacement for bin().
"""
import operator
import numpy as np

from unstruct import unstruct

class Placevalue(object):
    """
    Map integers to vectors [i0, i1, ...] with [n0, n1, ...] possible values
    
    Default endianness is most significant digit first, like binary numbers.
    Endianness affects the value of digit positions and conversion from 
    integer to vector. Conversion from vector to integer is always sum(v*posval)
    
    >>> p = Placevalue([4, 3, 2])
    >>> np.array([p.int2vec(i) for i in range(p.maxint)]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[0, 0, 0], [0, 0, 1], [0, 1, 0]...[3, 1, 1], [3, 2, 0], [3, 2, 1]])
    >>> b = Placevalue([2] * 8) # eight binary digits
    >>> b.int2vec(15)     # most significant digit first!
    array([0, 0, 0, 0, 1, 1, 1, 1])
    
    If a structured array is used to construct the Placevalue object, the same 
    dtype and field names are used for the vectors returned by int2vec(), 
    indexing or iterating.
    
    >>> dtype = [("a", np.int8), ("b", np.int8)]
    >>> pr = Placevalue(np.rec.fromrecords([[3, 4]], dtype=dtype))
    >>> pr[11]
    array([(2, 3)], dtype=[('a', '|i1'), ('b', '|i1')])
    
    This returns a structured ndarray rather than a recarray.
    If you prefer, you can:
    
    >>> pr[11].view(np.recarray)
    rec.array([(2, 3)], dtype=[('a', '|i1'), ('b', '|i1')])
    """
    
    def __init__(self, n, msd_first=True):
        """
        >>> p = Placevalue([4, 3, 2])
        >>> p.n
        array([4, 3, 2])
        >>> p.maxint
        24
        >>> p.posval
        array([6, 2, 1])
        >>> Placevalue([4, 3, 2], msd_first=False).posval
        array([ 1,  4, 12])
        """
        # Copy n into an array and make an unstructured view into it
        self.n = np.atleast_1d(n).copy()
        self.u = np.atleast_1d(np.squeeze(unstruct(self.n)))
        self.fieldtype = self.u.dtype
        self.dtype = self.n.dtype
        self.msd_first = msd_first
        # The number of possible genotypes can overflow fixed-size integers,
        # so use Python's unlimited-precision integer type.
        self.maxint = reduce(operator.mul, [int(i) for i in self.u])
        if msd_first:
            self.posval = np.r_[1, self.u[:0:-1].cumprod()][::-1]
        else:
            self.posval = np.r_[1, self.u[:-1].cumprod()]
    
    def __getitem__(self, i):
        """p[i] is a synonym for p.int2vec(i)."""
        return self.int2vec(i)
    
    def vec2int(self, v):
        """
        Integer corresponding to a vector [i0, i1, ...]
        
        The integer is computed by multiplying each element of i with its 
        place value and summing the result.
        
        >>> p = Placevalue([4, 3, 2])
        >>> p.vec2int([1, 2, 0])
        10
        >>> vmax = p.n - 1
        >>> vmax
        array([3, 2, 1])
        >>> p.vec2int(vmax) == p.maxint - 1 == 23
        True
        
        This function is vectorized:
        
        >>> arr = p.int2vec(range(3))
        >>> p.vec2int(arr)
        array([0, 1, 2])
        >>> arr = p.int2vec(range(p.maxint))
        >>> p.vec2int(arr.reshape(4,6,-1))
        array([[ 0,  1,  2,  3,  4,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17],
               [18, 19, 20, 21, 22, 23]])
        
        Structured dtype.
        
        >>> dtype = [("a", np.int8), ("b", np.int8)]
        >>> pr = Placevalue(np.rec.fromrecords([[3, 4]], dtype=dtype))
        >>> pr.vec2int(pr.int2vec(11))
        array([11])
        >>> pr.vec2int(np.concatenate(list(pr)))
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
        
        Range checking for digits.
        
        >>> p.vec2int([0, 0, 2])
        Traceback (most recent call last):
        ...
        OverflowError: Digit exceeds allowed range of [4 3 2]
        """
        # currently no support for len(v) < len(self.u)
        v = unstruct(v)
        if (v >= self.u).any():
            raise OverflowError("Digit exceeds allowed range of %s" % self.u)
        return (self.posval * v).sum(axis=v.ndim-1)
    
    def int2vec(self, i):
        """
        Vector [i0, i1, ...] corresponding to an integer.
        
        >>> p = Placevalue([4, 3, 2])
        >>> p.int2vec(13)
        array([2, 0, 1])
        >>> sum(p.posval * p.int2vec(13)) # analogous to 123 == 1*100+2*10+3*1
        13
        
        This function is vectorized:
        
        >>> p.int2vec(range(p.maxint)) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1],
        ...    [3, 0, 0], [3, 0, 1], [3, 1, 0], [3, 1, 1], [3, 2, 0], [3, 2, 1]])
        
        Compare with "least significant first" place values:
        
        >>> p = Placevalue([4, 3, 2], msd_first=False)
        >>> p.int2vec(range(p.maxint)) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], 
               [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0],
        ...    [0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1],
               [0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1]])
        
        Structured dtype.
        
        >>> dtype = [("a", np.int8), ("b", np.int8)]
        >>> pr = Placevalue(np.rec.fromrecords([[3, 4]], dtype=dtype))
        >>> pr.int2vec(11)
        array([(2, 3)], dtype=[('a', '|i1'), ('b', '|i1')])
        >>> np.concatenate(list(pr)) # doctest: +ELLIPSIS
        array([(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), ..., (2, 3)],
            dtype=[('a', '|i1'), ('b', '|i1')])
        
        Verify bugfix: If i is a sequence, the output should be a 2-d array 
        so that "for v in pv.int2vec(i)" works. Previously, this failed when 
        there was only one position.
        
        >>> a = Placevalue([3])
        >>> a.int2vec(range(4)) # doctest: +NORMALIZE_WHITESPACE
        array([[0], [1], [2], [3]])
        
        Bug if maxint exceeds the range of fixed-size integers.
        
        >>> a = Placevalue([[2] * 64])
        >>> v = [0] + [1] * 63
        >>> i = a.vec2int(v)
        >>> i
        9223372036854775807
        >>> a.int2vec(i) # doctest: +ELLIPSIS
        array([0, 1, 1, 1, 1, ...
        
        The actual result has overflowed:
        
        array([-1, -1,  1,  1,  1,  ...
        """
        i = np.atleast_1d(i).copy()
        result = np.zeros(i.shape + self.posval.shape, dtype=self.fieldtype)
        if self.msd_first:
            for pos, posval in enumerate(self.posval):
                result[:, pos], i = divmod(i, posval)
            result = result.squeeze()
        else:
            for pos, posval in enumerate(self.posval[::-1]):
                result[:, pos], i = divmod(i, posval)
            result = result[:, ::-1].squeeze()
        if len(self.u) == 1 and len(i) > 1:
            result = np.c_[result]
        return np.ascontiguousarray(result).view(self.dtype)
    
    def __repr__(self):
        """
        >>> Placevalue([4, 3, 2])
        Placevalue(array([4, 3, 2]), msd_first=True)
        
        >>> dtype = [("a", np.int8), ("b", np.int8)]
        >>> Placevalue(np.rec.fromrecords([[3, 4]], dtype=dtype))
        Placevalue(rec.array([(3, 4)],
              dtype=[('a', '|i1'), ('b', '|i1')]), msd_first=True)
        """
        return "%s(%r, msd_first=%r)" % (
            self.__class__.__name__, self.n, self.msd_first)
    
    def __len__(self):
        """
        Alias for Placevalue.maxint.
        
        Defining len() for Placevalue objects allows concatenation with 
        numpy.concatenate(). Se also __array__().
        
        >>> dtype = [("a", np.int8), ("b", np.int8)]
        >>> pv = Placevalue(np.rec.fromrecords([[2, 2]], dtype=dtype))
        >>> len(pv)
        4
        >>> np.concatenate(pv)
        array([(0, 0), (0, 1), (1, 0), (1, 1)], 
              dtype=[('a', '|i1'), ('b', '|i1')])
        
        Use maxint instead if __len__ exceeds the range of fixed-size integers.
        
        >>> len(Placevalue([[2] * 64]))
        Traceback (most recent call last):
        OverflowError: long int too large to convert to int
        >>> Placevalue([[2] * 64]).maxint
        18446744073709551616L
        """
        return self.maxint
    
    def __array__(self):
        """
        Return an array enumerating all vectors of a Placevalue object.
        
        Without named fields, this returns a 2-d ndarray.
        
        >>> np.array(Placevalue([2, 2]))
        array([[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])
        
        With named fields, this returns a 1-d recarray.
        
        >>> dtype = [("a", np.int8), ("b", np.int8)]
        >>> pv = Placevalue(np.rec.fromrecords([[2, 2]], dtype=dtype))
        >>> np.array(pv)
        array([(0, 0), (0, 1), (1, 0), (1, 1)],
              dtype=[('a', '|i1'), ('b', '|i1')])
        """
        if self.dtype.names: # structured ndarray
            return np.concatenate(self)
        else: # unstructured ndarray
            return np.vstack(self)  
    
    def __iter__(self):
        """
        Iterating over a Placevalue object returns successive vectors.
        
        >>> p = Placevalue([4, 3, 2])
        >>> np.array([i for i in p]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        array([[0, 0, 0], [0, 0, 1], [0, 1, 0]...[3, 1, 1], [3, 2, 0], [3, 2, 1]])
        """
        return (self.int2vec(i) for i in range(self.maxint))

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
        >>> np.concatenate(list(gt)) # doctest: +ELLIPSIS
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

def bin(i, ndigits=0):
    """
    >>> bin(13)
    '1101'
    >>> bin(13, 8)
    '00001101'
    >>> int(bin(13), 2)
    13
    """
    L = []
    digits = "01"
    while True:
        i, carry = divmod(i, 2)
        L.append(digits[carry])
        if i == 0:
            break
    L.reverse()
    return "".join(L).rjust(ndigits, "0")

def binarray(i, ndigits=0, dtype=int):
    """
    Numpy array of binary digits (most significant digit in position 0).
    
    >>> binarray(13)
    array([1, 1, 0, 1])
    >>> binarray(13, 8)
    array([0, 0, 0, 0, 1, 1, 0, 1])
    """
    return np.array(list(bin(i, ndigits)), dtype=dtype)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
