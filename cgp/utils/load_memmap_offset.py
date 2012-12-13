"""
Implementing offset and shape for io.load and format.open_memmap in numpy.lib.

Example using a Numpy array saved to a temporary directory.

>>> import tempfile, os, shutil
>>> dtemp = tempfile.mkdtemp()
>>> filename = os.path.join(dtemp, "test.npy")
>>> np.save(filename, np.arange(10))

>>> load(filename)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> mmap = load(filename, mmap_mode="r+")
>>> mmap
memmap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> mmap[3:7] = 42
>>> del mmap
>>> np.load(filename)
array([ 0,  1,  2, 42, 42, 42, 42,  7,  8,  9])
>>> mmap = load(filename, mmap_mode="r+", offset=2, shape=6)
>>> mmap[-1] = 123
>>> del mmap
>>> np.load(filename)
array([  0,   1,   2,  42,  42,  42,  42, 123,   8,   9])

For loading memmaps, shape and offset apply to the first dimension only;
the remaining dimensions are read from the file.

>>> x = np.arange(24.0).view([("a", float), ("b", float)]).reshape(4, 3)
>>> np.save(filename, x)

Each sub-array has three items. This skips sub-array 0, 
then extracts sub-arrays 1 and 2.

>>> load(filename, mmap_mode="r+", offset=1, shape=2)
memmap([[(6.0, 7.0), (8.0, 9.0), (10.0, 11.0)],
        [(12.0, 13.0), (14.0, 15.0), (16.0, 17.0)]], 
        dtype=[('a', '<f8'), ('b', '<f8')])
>>> shutil.rmtree(dtemp)
"""
import numpy as np
_file = file  # Hack borrowed from Numpy 1.4.0 np.lib.io
from numpy.lib.format import magic, read_magic, dtype_to_descr
from numpy.lib.format import read_array_header_1_0, write_array_header_1_0

def memmap_chunk_ind(filename, indices, mode="r+", check_contiguous=True):
    """
    Return a read-write memmap array containing given elements, and the offset.
    
    See memmap_chunk() if you just want chunk a.ID out of a.get_NID(), 
    where a is module "arrayjob".
    
    Example using Numpy array saved to a temporary directory.
    
    >>> import tempfile, os, shutil
    >>> dtemp = tempfile.mkdtemp()
    >>> filename = os.path.join(dtemp, "test.npy")
    >>> np.save(filename, np.arange(5))
    
    Typical usage.
    
    >>> memmap_chunk_ind(filename, range(2, 4))
    (memmap([2, 3]), 2)
    
    By default indices must make up a contiguous range (not necessarily sorted).
    
    >>> memmap_chunk_ind(filename, (1, 3, 4))
    Traceback (most recent call last):
    AssertionError: Indices not contiguous
    
    This skips the contiguity check.
    
    >>> memmap_chunk_ind(filename, (1, 3, 4), check_contiguous=False) 
    (memmap([1, 2, 3, 4]), 1)
    
    Note that the returned memmap has elements in the original order. 
    
    >>> ix = 1, 3, 2, 4
    >>> x, offset = memmap_chunk_ind(filename, ix)
    >>> x, offset
    (memmap([1, 2, 3, 4]), 1)
    
    Typical usage of the latter example:
    
    >>> [x[i - offset] for i in ix]
    [1, 3, 2, 4]
    
    Clean up after doctest.
    
    >>> del x
    >>> shutil.rmtree(dtemp)
    """
    indices = np.atleast_1d(indices)
    isort = sorted(indices)
    offset = isort[0]
    shape = 1 + isort[-1] - offset
    if check_contiguous:
        want = np.arange(offset, 1 + isort[-1])
        if (len(indices) != shape) or not all(isort == want):
            raise AssertionError, "Indices not contiguous"
    mm = open_memmap(filename, mode=mode, offset=offset, shape=shape)
    return mm, offset

def open_memmap(filename, mode='r+', dtype=None, shape=None,
                fortran_order=False, version=(1,0), offset=0):
    """
    Open a .npy file as a memory-mapped array, with offset argument.

    This may be used to read an existing file or create a new one.
    
    :param str filename: The name of the file on disk. This may not be a 
        file-like object.
    :param str mode: The mode to open the file with. In addition to the 
        standard file modes, 'c' is also accepted to mean "copy on write". 
        See `numpy.memmap` for the available mode strings.
    :param dtype dtype: The data type of the array if we are creating a 
        new file in "write" mode.
    :param tuple shape: The shape of the array if we are creating a new 
        file in "write" mode. Shape of (contiguous) slice if opening an 
        existing file.
    :param bool fortran_order: Whether the array should be Fortran-contiguous 
        (True) or C-contiguous (False) if we are creating a new file in 
        "write" mode.
    :param tuple version: If the mode is a "write" mode, then this is the 
        version (major, minor) of the file format used to create the file.
    :param int offset: Number of elements to skip along the first dimension.
    :return numpy.memmap: The memory-mapped array.

    Raises:
    
    * :exc:`ValueError` if the data or the mode is invalid
    * :exc:`IOError` if the file is not found or cannot be opened correctly.
    
    .. seealso:: :func:`numpy.memmap`
    """
    if not isinstance(filename, basestring):
        raise ValueError("Filename must be a string.  Memmap cannot use" \
                         " existing file handles.")

    if 'w' in mode:
        assert offset == 0, "Cannot specify offset when creating memmap"
        # We are creating the file, not reading it.
        # Check if we ought to create the file.
        if version != (1, 0):
            msg = "only support version (1,0) of file format, not %r"
            raise ValueError(msg % (version,))
        # Ensure that the given dtype is an authentic dtype object rather than
        # just something that can be interpreted as a dtype object.
        dtype = np.dtype(dtype)
        if dtype.hasobject:
            msg = "Array can't be memory-mapped: Python objects in dtype."
            raise ValueError(msg)
        d = dict(
            descr=dtype_to_descr(dtype),
            fortran_order=fortran_order,
            shape=shape,
        )
        # If we got here, then it should be safe to create the file.
        fp = open(filename, mode+'b')
        try:
            fp.write(magic(*version))
            write_array_header_1_0(fp, d)
            offset = fp.tell()
        finally:
            fp.close()
    else:
        # Read the header of the file first.
        fp = open(filename, 'rb')
        try:
            version = read_magic(fp)
            if version != (1, 0):
                msg = "only support version (1,0) of file format, not %r"
                raise ValueError(msg % (version,))
            fullshape, fortran_order, dtype = read_array_header_1_0(fp)
            
            if shape:
                length = np.atleast_1d(shape)
                msg = "Specify shape along first dimension only"
                assert length.ndim == 1, msg
            else:
                length = fullshape[0] - offset
            shape = (length,) + fullshape[1:]
            
            if dtype.hasobject:
                msg = "Array can't be memory-mapped: Python objects in dtype."
                raise ValueError(msg)
            
            offset_items = offset * np.prod(fullshape[1:], dtype=int)
            offset_bytes = fp.tell() + offset_items * dtype.itemsize
        finally:
            fp.close()
    
    if fortran_order:
        order = 'F'
    else:
        order = 'C'

    # We need to change a write-only mode to a read-write mode since we've
    # already written data to the file.
    if mode == 'w+':
        mode = 'r+'

    marray = np.memmap(filename, dtype=dtype, shape=shape, order=order,
        mode=mode, offset=offset_bytes)

    return marray

# pylint: disable=W0622
def load(file, mmap_mode=None, offset=0, shape=None):  #@ReservedAssignment
    """
    Load a pickled, ``.npy``, or ``.npz`` binary file.

    :param file file: The file to read. It must support ``seek()`` and 
        ``read()`` methods. If the filename extension is ``.gz``, the file is 
        first decompressed.
    :param str mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
        If not None, then memory-map the file, using the given mode
        (see `numpy.memmap`).  The mode has no effect for pickled or
        zipped files.
        
        A memory-mapped array is stored on disk, and not directly loaded
        into memory.  However, it can be accessed and sliced like any
        ndarray.  Memory mapping is especially useful for accessing
        small fragments of large files without reading the entire file
        into memory.
    :return: array, tuple, dict, etc. data stored in the file.

    .. seealso::
       
       save, savez, loadtxt
       memmap : Create a memory-map to an array stored in a file on disk.

    .. note::
    
       * If the file contains pickle data, then whatever is stored in the
         pickle is returned.
       * If the file is a ``.npy`` file, then an array is returned.
       * If the file is a ``.npz`` file, then a dictionary-like object is
         returned, containing ``{filename: array}`` key-value pairs, one for
         each file in the archive.
    
    Examples:
    
    Store data to disk, and load it again:

    >>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]])) # doctest: +SKIP
    >>> np.load('/tmp/123.npy') # doctest: +SKIP
    array([[1, 2, 3],
           [4, 5, 6]])

    Mem-map the stored array, and then access the second row
    directly from disk:

    >>> X = np.load('/tmp/123.npy', mmap_mode='r') # doctest: +SKIP
    >>> X[1, :] # doctest: +SKIP
    memmap([4, 5, 6])
    """
    if (not mmap_mode) and (offset or shape):
        raise ValueError("Offset and shape should be used only with mmap_mode")

    import gzip

    if isinstance(file, basestring):
        fid = _file(file, "rb")
    elif isinstance(file, gzip.GzipFile):
        fid = np.lib.npyio.seek_gzip_factory(file)
    else:
        fid = file

    # Code to distinguish from NumPy binary files and pickles.
    _ZIP_PREFIX = 'PK\x03\x04'
    N = len(np.lib.format.MAGIC_PREFIX)
    magic_ = fid.read(N)
    fid.seek(-N, 1) # back-up
    if magic_.startswith(_ZIP_PREFIX):  # zip-file (assume .npz)
        return np.lib.npyio.NpzFile(fid)
    elif magic_ == np.lib.format.MAGIC_PREFIX: # .npy file
        if mmap_mode:
            return open_memmap(file, mode=mmap_mode, shape=shape, offset=offset)
        else:
            return np.lib.format.read_array(fid)
    else:  # Try a pickle
        try:
            return np.lib.npyio._cload(fid)  # pylint: disable=W0212
        except:
            raise IOError, \
                "Failed to interpret file %s as a pickle" % repr(file)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
