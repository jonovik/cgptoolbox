"""open(filename, "w") only if the file does not exist."""

from contextlib import contextmanager
import os

@contextmanager
def write_if_not_exists(filename, raise_if_exists=False):
    """
    Context manager to open(filename, "w") only if the file does not exist.
    
    Usage:
        with write_if_not_exists(filename) as f:
            # write stuff to file
    
    If the file exists, f opens os.devnull instead of filename.
    Setting the optional keyword argument raise_if_exists=True will 
    raise OSError if the file already exists.
    
    Further usage examples, writing to a temporary directory.
    
    >>> from tempfile import mkdtemp
    >>> from shutil import rmtree
    >>> dtemp = mkdtemp()
    >>> filename = os.path.join(dtemp, "test.txt")
    
    Writing to a file that does not already exist. This works as normal.
    
    >>> with write_if_not_exists(filename) as f:
    ...     f.write("Hello world")
    >>> with open(filename) as f:
    ...     f.read()
    'Hello world'
    
    Writing again is silently ignored because raise_if_exists defaults to False.
    
    >>> with write_if_not_exists(filename) as f:
    ...     f.write("filename exists, so this goes to the null device")
    
    The file contents are unaffected.
    
    >>> with open(filename) as f:
    ...     f.read()
    'Hello world'
    
    Raising an exception if the file already exists.
    
    >>> with write_if_not_exists(filename, raise_if_exists=True):
    ...     f.write("This raises an exception") # doctest: +ELLIPSIS
    Traceback (most recent call last):
    OSError: File exists: ...test.txt
    
    Again, the file contents are unaffected.
    
    >>> with open(filename) as f:
    ...     f.read()
    'Hello world'
    
    >>> rmtree(dtemp) # cleanup after doctests
    """
    filename = os.path.realpath(filename) # os.makedirs() can't handle os.pardir
    dirname, _ = os.path.split(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if os.path.exists(filename):
        if raise_if_exists:
            raise OSError("File exists: %s" % filename)
        else:
            filename = os.devnull
    with open(filename, "w") as f:
        yield f

if __name__ == "__main__":
    import doctest
    doctest.testmod()
