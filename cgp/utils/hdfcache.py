"""
Function decorator to log/cache input/output recarrays to HDF group of tables

TODO: hdfmerge() WILL PROBABLY MESS UP THE ORDER OF RECORDS WITHIN EACH HDF 
FILE, BECAUSE IT TRIES TO SORT ALL TABLES. WHAT WE REALLY WANT IS TO SPECIFY A
KEY TO SORT BY. ARGH. MAYBE WE SHOULD JUST CONCATENATE.

Given a function func(input) -> output whose input and output are numpy record 
arrays, so that input.dtype.names and output.dtype.names are not None.

This module provides a way to automatically store input and output to 
HDF tables, using the dtype as table descriptors. Previously computed outputs 
are looked up rather than recomputed. Lookup is based on a the built-in hash()
applied to autoname(input).data
Gael Varoquaux' joblib offers another, slower, hash for Numpy arrays, which is 
probably not needed for our purposes.
https://launchpad.net/joblib

Several functions can be cached to the same HDF file. To achieve this, the 
decorator is a method of an object that owns the file. (This design is borrowed 
from Gael Varoquaux's "joblib" package, https://launchpad.net/joblib.)

@todo: log process id or job id
@todo: Allow read-only opening of a cache. This would require parsing options 
at the start of the module.
@todo: Use temp file for doctests by default, simplifying the following code: 

An example follows, using a temporary file by default. If you'd like to inspect 
the HDF output, you can specify a filename by passing a --filename option when 
running the doctests.

>>> try: filename = options.filename
... except NameError: filename = raw_input("Name of output file (default=''): ")
>>> if not filename:
...     import tempfile
...     filename = os.path.join(tempfile.mkdtemp(), 'cachetest.h5')

First, create an Hdfcache instance.

>>> hdfcache = Hdfcache(filename)

Now its method hdfcache.cache(func) can be used as a decorator:

>>> @hdfcache.cache
... def f(x, a, b=10):
...     "This is the docstring for function f"
...     result = np.zeros(1, dtype=[("y", float)])
...     result["y"] = x["i"] * 2 + x["f"] + a * b
...     print "Evaluating f:", x, a, b, "=>", result
...     return result

Calling hdfcache.cache again will cache multiple functions in the same HDF file.

>>> @hdfcache.cache
... def g(y):
...     "This is the docstring for the function g"
...     result = np.zeros(1, dtype=[("a", np.int32), ("b", float), ("c", complex)])
...     result["a"] = int(y["y"])
...     result["b"] = 1 / y["y"]
...     result["c"] = complex(0, y["y"])
...     print "Evaluating g:", y, "=>", result
...     return result

The function is evaluated normally the first time an input is encountered.

>>> x = np.rec.fromarrays([[1, 2], [1.25, 3.5]], dtype=[("i", np.int32), ("f", float)])
>>> np.concatenate([g(f(xi, 5)) for xi in x])
Evaluating f: [(1, 1.25)] 5 10 => [(53.25,)]
Evaluating g: [(53.25,)] => [(53, 0.0187793..., 53.25j)]
Evaluating f: [(2, 3.5)] 5 10 => [(57.5,)]
Evaluating g: [(57.5,)] => [(57, 0.0173913..., 57.5j)]
array([(53, 0.0187793..., 53.25j),
       (57, 0.0173913..., 57.5j)],
      dtype=[('a', '<i4'), ('b', '<f8'), ('c', '<c16')])

In addition, the input and output are stored in the HDF file.

>>> print "Cache:", hdfcache.file    # doctest output cannot start with ellipsis
Cache: ...
/f (Group) ''
/f/hash (Table(2,), shuffle, zlib(1)) ''
/f/input (Table(2,), shuffle, zlib(1)) ''
/f/output (Table(2,), shuffle, zlib(1)) ''
/f/timing (Table(2,), shuffle, zlib(1)) ''
/g (Group) ''
/g/hash (Table(2,), shuffle, zlib(1)) ''
/g/input (Table(2,), shuffle, zlib(1)) ''
/g/output (Table(2,), shuffle, zlib(1)) ''
/g/timing (Table(2,), shuffle, zlib(1)) ''

Calling the decorated function again with the same inputs will avoid calling 
the original function.

>>> f(x[1], 5)
rec.array([(57.5,)], dtype=[('y', '<f8')])

The source code of a cached function is stored as an attribute of its Group.

>>> print hdfcache.file.root.f._v_attrs.sourcecode
@hdfcache.cache
def f(x, a, b=10):...
>>> hdfcache.file.root.f._v_attrs
/f._v_attrs (AttributeSet),...
    sourcecode := '...',
    sourcefile := '<doctest __main__[3]>']

Closing the HDF file:
hdfcache.file is a property that will create the file if needed and reopen it 
if it has been closed. Remember to close the file, either explicitly:

>>> hdfcache.file.close()
>>> hdfcache._file # The Hdfcache's internal tables.File instance
<closed File>

or using a "with" statement:

>>> with hdfcache:
...     hdfcache.file
File(filename=..., title='', mode='a', rootUEP='/', ...
>>> hdfcache._file
<closed File>

Compression is enabled by default, but can be disabled by passing filters=None 
to the Hdfcache() constructor.


== Comparing hash values manually ==

The ahash(x) function currently applies the built-in hash() to x.__data__, 
converting x to a Numpy array if needed. This is fast and tolerant towards 
minor differences in shape and dtype. Just be aware of cases like these, which 
have the same binary __data__:

>>> ahash(np.zeros(1)) == ahash(np.zeros(8, np.int8))       # float is 8 bytes
True

Here follow some details that are not relevant to the current ahash(), but are 
important if you choose a stricter hash function.

Numpy structured arrays and PyTables tables both distinguish between:

>>> type(x[0])
<class 'numpy.core.records.record'>
>>> type(x[0:1])
<class 'numpy.core.records.recarray'>
>>> x[0]
(1, 1.25)
>>> x[0:1]
rec.array([(1, 1.25)], dtype=[('i', '<i4'), ('f', '<f8')])

Note, though, that 

>>> x[0].dtype == x[0:1].dtype
True

These nuances may affect hash values: whether an input is ndarray or recarray, 
its shape and dimensionality.

>>> ahash(x[1]) == ahash(x[1:2]) # False if using joblib.hash as hash
True

The following is a reliable way to check whether 
the hash of an input is in an existing hash Table.
We are searching for record x[1] in the HDF group hdfcache.file.root.f.

>>> with hdfcache.file as f:
...     f.root.f.hash.getWhereList("hash == h", dict(h=ahash(autoname(x[1]))))
array([1]...)

Details:

>>> h1 = ahash(x[1:2])
>>> h1      # with joblib.hash: '4a14c07534c7a24053e58540b74de973'
... # on Windows XP 32-bit: 1710834134
6484299406236008918
>>> with hdfcache:
...     hash = hdfcache.file.root.f.hash # Table object
...     h = hash[:] # extract all records as structured ndarray
>>> h # with joblib.hash: array([('cadf9e2413df9d83e2303522bc1267a9',), ('4a14c07534c7a24053e58540b74de973',)], dtype=[('hash', '|S32')])
... # on Windows XP 32-bit: array([(773416804,), (1710834134,)], dtype=[('hash', '<i4')])
array([(-5981222333818771612,), (6484299406236008918,)], dtype=[('hash', '<i8')])
>>> np.where(h1 == h["hash"])
(array([1]),)

These are equivalent:

>>> ahash(x[1:2]) == ahash(autoname(x[1]))
True

For the painful details, see 
http://thread.gmane.org/gmane.comp.python.pytables.user/1238/focus=1250
"""

# for merging iterators
import heapq
import itertools
from contextlib import nested
from glob import glob
import os
from poormanslock import Lock
import shutil
# for handling numpy record arrays and HDF tables
import tables as pt
import numpy as np
from argrec import autoname
# for decorating
import inspect
from functools import wraps
import time
# logging facilities, useful for debugging
import logging 

# # To use Gael Varoquaux' joblib.hash() rather than the built-in one.
# try:
#     from joblib import hash as ahash
# except ImportError:
#     pass


log = logging.getLogger("hdfcache")
log.addHandler(logging.StreamHandler())
# tab-delimited format string, see http://docs.python.org/library/logging.html#formatter-objects
fmtstr = "%(" + ")s\t%(".join(
    "asctime levelname name lineno process message".split()) + ")s"
log.handlers[0].setFormatter(logging.Formatter(fmtstr))


def ahash(x):
    """
    Hash the raw data of a Numpy array
    
    The input will be converted to a Numpy array if possible.    
    The shape and dtype of the array does not enter into the hash.
    
    >>> x = np.arange(5)
    >>> y = np.arange(5)
    >>> ahash(x) == ahash(y) == ahash(range(5))
    True
    """
    x = np.asarray(x)
    x.setflags(write=False)
    return hash(x.data)


class NoHdfcache(object):
    """A caricature of a caching decorator that does not actually cache"""
    
    def __init__(self, filename):
        """Initialize resources shared among caches"""
        pass
    
    def cache(self, func):
        """
        A null decorator implemented as an instance method

        >>> nohdfcache = NoHdfcache("dummy.filename")
        >>> @nohdfcache.cache
        ... def f(x): return x * x
        >>> @nohdfcache.cache
        ... def g(y): return y * y * y
        >>> f(2)
        4
        >>> g(3)
        27
        """
        return func


class DictHdfcache(object):
    """Prototype of caching using a shared resource (a dict)"""
    
    def __init__(self, filename):
        """Initialize a single dict that will hold multiple caches"""
        self.d = {} # owns all the caches, similar to a HDF5 File object
        self.argspec = {} # prototype for making "args" Table
        self.output_type = {} # deferring details of storage to later
    
    def cache(self, func):
        """
        Cache a function, using a resource in scope of an object instance
        
        Note that "self" is in scope of the decorator method, providing access 
        to shared resources. By contrast, the scope of "func" is limited to 
        each decoration. Each decoration involves a single call to cache().
        It initializes any sub-resources specific to "func".
        Finally, it defines the actual wrapper as a plain function, but one 
        whose scope includes both "self" and "func".
        
        Access to "self" preserves information (the cache) between calls to the 
        wrapped function. Because it is a plain function, it can adopt the 
        docstring of func by use of @wraps(func).
        
        Details of the caching can be deferred until the required information 
        is available, like knowing the dtype of an output array before creating 
        a corresponding HDF table.
        
        Verifying what's going on...
        
        >>> dicthdfcache = DictHdfcache("dummy.filename")
        >>> @dicthdfcache.cache
        ... def f(x): return x * x
        cache: initializing resources for a decorated function
        >>> f(2)
        cache: computed f 2 => 4
        cache: deferred initialization
        4
        >>> f(3)
        cache: computed f 3 => 9
        9
        >>> f(3)
        cache: returning cached value
        9
        
        Create another cache using the same resource (a dict); the end result 
        is shown below.
        
        >>> @dicthdfcache.cache
        ... def g(y): return y * y * y
        cache: initializing resources for a decorated function
        >>> g(3)
        cache: computed g 3 => 27
        cache: deferred initialization
        27
        
        Here's the dictionary with the two caches.
        
        >>> dicthdfcache.d # with joblib.hash: {<function f at 0x...>: {'bd344c9d...': 9, '0be9508f...': 4}, <function g at 0x...>: {'bd344c9d...': 27}}
        ... # on Windows XP 32-bit: {<function f at 0x015E1D30>: {921594546: 4, -765091819: 9}, <function g at 0x015E1EB0>: {-765091819: 27}}
        {<function f at 0x...>: {9147803436690843753: 9, -6199293758012471906: 4}, 
         <function g at 0x...>: {9147803436690843753: 27}}
        
        A function with both required, default, and variable-length unnamed and 
        keyword arguments.
        
        >>> @dicthdfcache.cache
        ... def h(a, b=10, *args, **kwargs): pass
        cache: initializing resources for a decorated function
        
        Here are the argument specifications, which could be used for deferred
        specification of an "args" table.
        
        >>> dicthdfcache.argspec
        {<function f at 0x...>: ArgSpec(args=['x'], 
        varargs=None, keywords=None, defaults=None), 
        <function g at 0x...>: ArgSpec(args=['y'], 
        varargs=None, keywords=None, defaults=None), 
        <function h at 0x...>: ArgSpec(args=['a', 'b'],
        varargs='args', keywords='kwargs', defaults=(10,))}
        """
        print "cache: initializing resources for a decorated function"
        self.d[func] = {} # nested dictionary, like a cache Group in a File
        self.argspec[func] = inspect.getargspec(func)
        
        @wraps(func)
        def wrapper(input):
            # ignoring the complications of hashing multiple arguments
            key = ahash(input)
            if key in self.d[func]:
                print "cache: returning cached value"
                return self.d[func][key]
            else:
                output = func(input)
                print "cache: computed", func.__name__, input, "=>", output
                if not self.d[func]:
                    # If the func-specific nested dict is empty, we know that 
                    # this is the first time func is evaluated. Now we can 
                    # perform initialization that had to be deferred until 
                    # we knew what kind of output func produces.
                    print "cache: deferred initialization"
                    self.output_type[func] = type(output) # just an example
                self.d[func][key] = output # store value for later retrieval
                return output
        
        return wrapper


class Hdfcache(object):
    """HDF file wrapper with function caching decorator"""
    
    def __init__(self, filename, where="/", filters=pt.Filters(complevel=1), 
                 mode="a", withflagfile=True, 
                 *args, **kwargs):
        """
        Constructor for HDF cache object.
        
        Arguments "filename", "filters", "mode" are passed to Tables.openFile().
        Argument "where" identifies a parent group for all the function caches.
        The boolean argument "withflagfile" says whether to create a flag file 
        with the extension ".delete_me_to_stop" that indicates that the process 
        is running. Deleting or renaming that file will raise an exception at a 
        time when no function is being evaluated, ensuring clean exit and 
        flushing of buffers.
        """
        kwargs["filename"] = filename
        kwargs["mode"] = mode
        kwargs["filters"] = filters
        self._file = None
        self.where = where
        self.fileargs = args
        self.filekwargs = kwargs
        self.withflagfile = withflagfile
        if withflagfile:
            self.flagfilename = filename + ".delete_me_to_stop"
            self.incontext = False
    @property
    def file(self):
        """
        File object for an HDF cache, see Tables.File in PyTables.
        
        The file is created if it doesn't exist, reopened if it has been closed.
        """
        if not (self._file and self._file.isopen):
            self._file = pt.openFile(*self.fileargs, **self.filekwargs)
            log.debug("Opened cache file")
        return self._file
    
    def group(self, funcname):
        """Dictionary of HDF parent groups for each function."""
        try:
            return self.file.getNode(self.where, funcname)
        except pt.NoSuchNodeError:
            return self.file.createGroup(self.where, funcname, createparents=True)
    
    def __enter__(self):
        """
        Enter the context of a with statement, optionally creating flag file.
        
        This doctest tests the flag file functionality. The flag file is 
        deleted on the first pass through the function, causing a 
        KeyboardInterrupt. (KeyboardInterrupt is not caught by doctest, hence 
        the explicit try..finally.)
        
        >>> import tempfile, os
        >>> filename = os.path.join(tempfile.mkdtemp(), 'entertest.h5')
        >>> cacher = Hdfcache(filename)
        >>> @cacher.cache
        ... def f(x):
        ...     os.remove(cacher.flagfilename)
        ...     return x
        >>> try:
        ...     with cacher:
        ...         while True:
        ...             y = f(0)
        ... except KeyboardInterrupt:
        ...     print "Flag file not found."
        Flag file not found.
        """
        if self.withflagfile:
            self.incontext = True
            open(self.flagfilename, "w").close() # create empty file
        return self
    
    def __exit__(self, type, value, tb):
        """
        Exit context of with statement, closing file and removing any flag file.
        """
        if self._file and self._file.isopen:
            self.file.close()
        if self.withflagfile and os.path.exists(self.flagfilename):
            self.incontext = False
            os.remove(self.flagfilename)
    
    def cache(self, func):
        """
        Decorator for function-specific caching of inputs and outputs
        
        This gets called once for each function being decorated, creating a 
        new scope with HDF node objects specific to the decorated function.
        
        The Group object of the decorated function will currently not survive 
        after the first with statement, because the File is closed and the 
        node initialization code is never called again. I'll need to either put 
        all of the "open if exist else create" stuff in the wrapper below, or 
        encapsulate that into a separate object, which has to know about both 
        file (in the scope of the "hdfcache" instance) and 
        func (in the scope of the "cache" function).
        
        How to tell if a node is closed: _v_isopen
        Natural naming will open if required, so just need the group and always 
        be explicit about group.hash, group.input, group.output.
        
        @todo: make iterator that buffers hashes so we can read many at a time 
        using table.itersequence, see: 
        http://wiki.umb.no/CompBio/index.php/HDF5_in_Matlab_and_Python
        """
        funcname = func.__name__
        group = self.group(funcname) # reopening file if required
        self.set_source_attr(group, func)
        hashdict = dict(uninitialized=True)
        
        @wraps(func)
        def wrapper(input, *args, **kwargs):
            if self.withflagfile and self.incontext:
                if not os.path.exists(self.flagfilename):
                    log.debug("Flag file not found when calling %s", func)
                    raise KeyboardInterrupt # will flush cache on __exit__
            input = autoname(input)
            ihash = ahash(input)
            group = self.group(funcname) # reopening file if required
            if "uninitialized" in hashdict:
                # Load the hash table, creating it if necessary
                try:
                    hash = group.hash
                    log.debug("Reading existing hashes")
                    # Pitfall: Iterating over the hash Table may return 
                    # objects that are not strings and therefore will never 
                    # match the input hash.
                    # See http://osdir.com/ml/python.pytables.user/2007-12/msg00002.html
                    # When iterating over the hash Table to compare hashes, 
                    # the hash value must be extracted from each record.
                    # Iterating over the hash Table itself yields 
                    # a Row instance with a single field called "_0". Thus:
                    # [row["_0"] for row in hash]
                    #   is the desired list of strings.
                    # [row for row in hash]
                    #   is a list of multiple copies of the last row.
                    # hash[:] is a 1-d recarray of strings, which iterates to 
                    #   tuples, so  
                    # [h for (h,) in hash[:]] is the desired list of strings.
                    hashdict.update((h, i) for i, (h,) in enumerate(hash[:]))
                except pt.NoSuchNodeError:
                    log.debug("Creating hash table for %s", func)
                    hashdescr = autoname(ihash)[:0]
                    hashdescr.dtype.names = ["hash"]
                    hash = self.file.createTable(group, "hash", hashdescr)
                    self.set_source_attr(hash, ahash)
                del hashdict["uninitialized"]
            if ihash in hashdict:
                log.debug("Cache hit %s: %s %s", func, ihash, input)
                # Prevent ugly "ValueError: 0-d arrays can't be concatenated" 
                # http://projects.scipy.org/numpy/wiki/ZeroRankArray
                return autoname(group.output[hashdict[ihash]])
            else:
                log.debug("Cache miss %s: %s %s", func, ihash, input)
                timing = np.rec.fromarrays([[0.0], [0.0], [0.0]], 
                                           names=["seconds", "start", "end"])
                timing.start = time.clock()
                output = autoname(func(input, *args, **kwargs))
                timing.end = time.clock()
                timing.seconds = timing.end - timing.start
                if hashdict: # tables exist, but no record yet for this input
                    hash = group.hash
                    log.debug("Appending to input, output, and timing tables")
                    group.input.append(input)
                    group.output.append(output)
                    group.timing.append(timing)
                else: # make tables from recarray descriptor, store first record
                    log.debug("Creating input, output, and timing tables")
                    self.file.createTable(group, "input", input)
                    self.file.createTable(group, "output", output)
                    self.file.createTable(group, "timing", timing)
                hashdict[ihash] = hash.nrows
                hash.append(autoname(ihash))
                return output
        
        # close the file so the decorator doesn't require a "with" statement
        self.file.close()
        return wrapper
    
    @staticmethod
    def set_source_attr(node, obj):
        """Store the source code of an object as an attribute of an HDF node."""
        if "sourcecode" not in node._v_attrs:
            try:
                node._v_attrs.sourcefile = inspect.getfile(obj)
                node._v_attrs.sourcecode = inspect.getsource(obj)
            except (TypeError, IOError):
                node._v_attrs.sourcefile = "built-in"
                node._v_attrs.sourcecode = ""


def hdfmerge(pathname="*.h5", outfilename="merged.h5"):
    msg = """
        Use hdfcat() instead, because sorting is tricky and it is not obvious 
        what to sort by. 
    """
    raise NotImplementedError(msg)

def _hdfmerge(pathname="*.h5", outfilename="merged.h5"):
    """
    (Abandoned code.) Merge and sort data scattered over many HDF files with equal layout.
        
    All HDF files matching pathname are merged into a new file denoted by 
    outfilename. If the output file already exists, no action is taken. The 
    function returns True if the output file was created and False otherwise.
    A lock is held while merging, so that work can be shared between multiple 
    instances of a script (see the "grabcounter" module), while only one 
    instance merges the results.
    
    The use of iterators conserves memory, so this should work for arbitrarily 
    large data sets. It is also quite IO-efficient (at least up to 300 files 
    totalling 10 MB...haven't tested more yet); merging takes only fractionally 
    longer than the original writing.
    The only limitation is perhaps that we need simultaneous handles to all 
    input files. Also, this currently only merges tables, not arrays. (It might 
    work out of the box, at least for VLArray.)
    
    Todo: Guard against adopting the structure of an unpopulated cache file 
    left by job instances that arrived too late to do any work. Currently I use 
    the biggest file and hope that's okay.
    
    Adapted from http://cilit.umb.no/WebSVN/wsvn/Cigene_Repository/CigeneCode/CompBio/cGPsandbox/h5merge.py
    
    The following doctests are more for testing than documentation.
    
    Distribute sample data over three HDF files in a temporary directory.
    
    >>> import tempfile
    >>> filename = os.path.join(tempfile.mkdtemp(), 'cachetest.h5')
    >>> a = np.rec.fromarrays([[0, 2, 1]], names="a")
    >>> b = np.rec.fromarrays([[11, 12, 10]], names="b")
    >>> def writedata(i):
    ...     with pt.openFile("%s.%s.h5" % (filename, i), "w") as f:
    ...         f.createTable(f.root, "a", a[i:i+1])
    ...         f.createTable("/group1", "b", b[i:i+1], createparents=True)
    ...         return str(f.root.a[:]) + " " + str(f.root.group1.b[:]) + " " + str(f)
    >>> for i in 0, 1, 2:
    ...     print "Part", i, writedata(i)
    Part 0 [(0,)] [(11,)] ...cachetest.h5.0.h5...
    / (RootGroup) ''
    /a (Table(1,)) ''
    /group1 (Group) ''
    /group1/b (Table(1,)) ''
    Part 1 [(2,)] [(12,)] ...cachetest.h5.1.h5...
    / (RootGroup) ''
    /a (Table(1,)) ''
    /group1 (Group) ''
    /group1/b (Table(1,)) ''
    Part 2 [(1,)] [(10,)] ...cachetest.h5.2.h5...
    / (RootGroup) ''
    /a (Table(1,)) ''
    /group1 (Group) ''
    /group1/b (Table(1,)) ''
    
    Merge them together.
    
    >>> hdfmerge(filename + ".*.h5", filename + ".merged") # doctest: +SKIP
    True
    >>> with pt.openFile(filename + ".merged") as f:
    ...     print "Merged", str(f)    # doctest: +SKIP
    Merged ...cachetest.h5.merged...
    / (RootGroup) ''
    /a (Table(3,), shuffle, zlib(1)) ''
    /group1 (Group) ''
    /group1/b (Table(3,), shuffle, zlib(1)) ''
    
    TODO: NOTE THAT EACH TABLE IS SORTED SEPARATELY. 
    WHAT I REALLY WANT IS TO SORT ALL TABLES BY ONE OF THEM.
    
    >>> with pt.openFile(filename + ".merged") as f:
    ...     print f.root.a[:], f.root.group1.b[:] # doctest: +SKIP
    [(0,) (1,) (2,)] [(11,) (10,) (12,)]

    False is returned if the output file already exists.
    
    >>> hdfmerge(filename + ".*.h5", filename + ".merged") # doctest: +SKIP
    False
    """
    try:
        with Lock(outfilename + ".lock"):
            if os.path.exists(outfilename):
                return False
            # Find names of all files to be merged, sort by file size.
            infilenames = glob(pathname)
            infilenames.sort(key=os.path.getsize, reverse=True)
            # Open them all safely, using "with nested()"
            with nested(*(pt.openFile(i) for i in infilenames)) as fin:
                # Clone the HDF structure of the biggest input file into the outfile 
                # (stop=0 omits all records), hoping that the biggest file has all 
                # the caches populated.
                fin[0].copyFile(outfilename, stop=0, filters=pt.Filters(complevel=1))
                with pt.openFile(outfilename, "a") as fout:
                    
                    def tablewalkers(f):
                        """List of iterators over nodes of HDF files."""
                        # tables.File.walkNodes is an iterator, here limited to Table's
                        return [fi.walkNodes(classname="Table") for fi in f]
                    
                    # Remove HDF files with no actual content.
                    # Exhaust all the iterators and see which ones end up None.
                    for t in itertools.izip_longest(*tablewalkers(fin)):
                        pass
                    fin = [fi for fi, ti in zip(fin, t) if ti]
                    
                    # izip yields one node per file at a time, 
                    # presumably in identical order.
                    # (The plus in "[fout] + fin" is list concatenation.) 
                    for t in itertools.izip(*tablewalkers([fout] + fin)):
                        tout, tin = t[0], t[1:]
                        # see "Mergesorting tables with heapq.merge",
                        # http://article.gmane.org/gmane.comp.python.pytables.user/1440
                        # [(row[:] for row in ti) for ti in tin] is a list with one 
                        # iterator per table/file.
                        for inrow in heapq.merge(*[(row[:] for row in ti) for ti in tin]):
                            # Table.append() needs a sequence, so wrap inrow in a list
                            tout.append([inrow])
        return True
    except IOError, exc:
        if "Timed out" in str(exc):
            return False
        else:
            raise

def hdfcat(pathname="*.h5", outfilename="concatenated.h5"):
    """
    Concatenate data scattered over many HDF files with equal layout.
    
    All HDF files matching pathname are concatenated into a new file denoted by 
    outfilename. If the output file already exists, no action is taken. The 
    function returns True if the output file was created and False otherwise.
    A lock is held while concatenating, so that work can be shared between 
    multiple instances of a script (see the "grabcounter" module), while only 
    one instance concatenates the results.
    
    Compression settings are inherited from the biggest file.
    
    NOTE: NEED TO ENSURE THAT ALL PROCESSES HAVE FINISHED FLUSHING HDF BUFFERS 
    BEFORE CONCATENATING. See grabcounter.grabsummer().
    
    The use of iterators conserves memory, so this should work for arbitrarily 
    large data sets. It is also quite IO-efficient due to PyTables' buffering 
    of Table objects. The only limitation is perhaps that we need simultaneous 
    handles to all input files. Also, this currently only concatenates tables, 
    not arrays. (It might work out of the box, at least for VLArray.)
    
    Todo: Guard against adopting the structure of an unpopulated cache file 
    left by job instances that arrived too late to do any work. Currently I use 
    the biggest file and hope that's okay.
    
    Adapted from http://cilit.umb.no/WebSVN/wsvn/Cigene_Repository/CigeneCode/CompBio/cGPsandbox/h5merge.py
    
    The following doctests are more for testing than documentation.
    
    Distribute sample data over three HDF files in a temporary directory.
    
    >>> import tempfile
    >>> filename = os.path.join(tempfile.mkdtemp(), 'cachetest.h5')
    >>> a = np.rec.fromarrays([[0, 2, 1]], names="a")
    >>> b = np.rec.fromarrays([[11, 12, 10]], names="b")
    >>> def writedata(i):
    ...     with pt.openFile("%s.%s.h5" % (filename, i), "w") as f:
    ...         f.createTable(f.root, "a", a[i:i+1])
    ...         f.createTable("/group1", "b", b[i:i+1], createparents=True)
    ...         return str(f.root.a[:]) + " " + str(f.root.group1.b[:]) + " " + str(f)
    >>> for i in 0, 1, 2:
    ...     print "Part", i, writedata(i)
    Part 0 [(0,)] [(11,)] ...cachetest.h5.0.h5...
    / (RootGroup) ''
    /a (Table(1,)) ''
    /group1 (Group) ''
    /group1/b (Table(1,)) ''
    Part 1 [(2,)] [(12,)] ...cachetest.h5.1.h5...
    / (RootGroup) ''
    /a (Table(1,)) ''
    /group1 (Group) ''
    /group1/b (Table(1,)) ''
    Part 2 [(1,)] [(10,)] ...cachetest.h5.2.h5...
    / (RootGroup) ''
    /a (Table(1,)) ''
    /group1 (Group) ''
    /group1/b (Table(1,)) ''

    Part 0 [(0, 11)] ...cachetest.h5.0.h5...
    / (RootGroup) ''
    /data (Table(1,)) ''
    /group1 (Group) ''
    /group1/data (Table(1,)) ''
    Part 1 [(2, 12)] ...cachetest.h5.1.h5...
    / (RootGroup) ''
    /data (Table(1,)) ''
    /group1 (Group) ''
    /group1/data (Table(1,)) ''
    Part 2 [(1, 10)] ...cachetest.h5.2.h5...
    / (RootGroup) ''
    /data (Table(1,)) ''
    /group1 (Group) ''
    /group1/data (Table(1,)) ''
    
    Concatenate them together.
    
    >>> hdfcat(filename + ".*.h5", filename + ".concatenated")
    True
    >>> with pt.openFile(filename + ".concatenated") as f:
    ...     print "Concatenated", str(f)
    Concatenated ...cachetest.h5.concatenated...
    / (RootGroup) ''
    /a (Table(3,)) ''
    /group1 (Group) ''
    /group1/b (Table(3,)) ''
    
    Note that the output is not sorted.
    
    >>> with pt.openFile(filename + ".concatenated") as f:
    ...     print f.root.a[:], f.root.group1.b[:]
    ... # on Windows XP 32-bit: [(1,) (0,) (2,)] [(10,) (11,) (12,)]
    [(2,) (1,) (0,)] [(12,) (10,) (11,)]
    

    False is returned if the output file already exists.
    
    >>> hdfcat(filename + ".*.h5", filename + ".concatenated")
    False
    """
    try:
        with Lock(outfilename + ".lock"):
            if os.path.exists(outfilename):
                return False
            # Find names of all files to be merged, sort by file size.
            infilenames = glob(pathname)
            infilenames.sort(key=os.path.getsize)
            bigfilename = infilenames.pop()
            # Open them all safely, using "with nested()"
            with nested(*(pt.openFile(i) for i in infilenames)) as fin:
                # Copy the biggest HDF file so we can append data 
                # from the others into the same structure. This relies on 
                # the biggest file having all the caches populated.
                # Don't use pt.copyFile() because it is slow on complex nested 
                # columns, http://www.pytables.org/trac/ticket/260
                shutil.copy(bigfilename, outfilename)
                with pt.openFile(outfilename, "a") as fout:
                    
                    def tablewalkers(f):
                        """List of iterators over nodes of HDF files."""
                        # tables.File.walkNodes is an iterator, here limited to Table's
                        return [fi.walkNodes(classname="Table") for fi in f]
                    
                    # Remove HDF files with no actual content.
                    # Exhaust all the iterators and see which ones end up None.
                    for t in itertools.izip_longest(*tablewalkers(fin)):
                        pass
                    fin = [fi for fi, ti in zip(fin, t) if ti]
                    
                    # izip yields one node per file at a time, 
                    # presumably in identical order.
                    # (The plus in "[fout] + fin" is list concatenation.) 
                    for t in itertools.izip(*tablewalkers([fout] + fin)):
                        tout, tin = t[0], t[1:]
                        for ti in tin:
                            tout.append(ti[:])
        return True
    except IOError, exc:
        if "Timed out" in str(exc):
            return False
        else:
            raise



if __name__ == "__main__":
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-o", "--filename", help="Name of output HDF5 file")
    parser.add_option("-v", "--verbose", action="store_true", 
        help="Run doctests with verbose output")
    parser.add_option("--debug", action="store_true", 
        help="Turn on debug logging for HDF cache")
    options, args = parser.parse_args()
    if options.debug:
        log.setLevel(logging.DEBUG)
    import doctest
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    doctest.testmod(optionflags=optionflags)
