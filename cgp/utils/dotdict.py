#!/usr/bin/env python
"""Dictionary allowing d.key = value"""

import pprint

class Dotdict(dict):
    """
    Dictionary allowing d.key = value
    
    A Dotdict uses the usual dict constructor.
    
    >>> d = Dotdict(a=1, b="test")
    
    Item access by dotting.
    
    >>> d.a
    1
    >>> d.b
    'test'
    
    Item assignment by dotting.
    
    >>> d.c = "new item"
    
    The usual dict features are there, e.g. equivalence.
    
    >>> d == {"a": 1, "b": "test", "c": "new item"}
    True
    
    
    Trying to access an undefined field.
    
    >>> "e" in d
    False
    >>> d.e
    Traceback (most recent call last):
    AttributeError: 'super' object has no attribute 'e'
    
    Buglet: The previous test should have raised KeyError: 'e', 
    but then pickling wouldn't work, see Dotdict.__getattr__().
    
    Not all valid dict keys can be used as attributes
    (eg. numbers and Python keywords). In this case, use regular dict syntax.
    
    >>> d.if = 0
    Traceback (most recent call last):
    SyntaxError: invalid syntax
    >>> d["if"] = 0
    >>> d.1 = 1
    Traceback (most recent call last):
    SyntaxError: invalid syntax
    >>> d[1] = 1
    
    Adding a field with the same name as a method or attribute of a dict.
    Assignment works, but access requires [brackets].
    
    >>> d.copy = 1
    >>> d.copy # doctest: +ELLIPSIS
    <built-in method copy of Dotdict object at 0x...>
    >>> d["copy"]
    1
    
    Incidentally, Dotdict.copy() returns a regular dict.
    
    >>> d # doctest: +ELLIPSIS
    Dotdict({...})
    >>> d.copy() # doctest: +ELLIPSIS
    {...}
    """
    def __getattr__(self, name):
        """
        Returns self[name] if available, otherwise call inherited __getattr__.
        
        >>> from cStringIO import StringIO
        >>> import numpy as np
        >>> f = StringIO()
        >>> d = Dotdict(a=1)
        >>> np.save(f, d)
        >>> f.reset()
        >>> np.load(f)
        array(Dotdict({'a': 1}), dtype=object)
        """
        try:
            return self[name]
        except KeyError:
            getattr(super(Dotdict, self), name)
    def __setattr__(self, name, value):
        self[name] = value
    def __repr__(self):
        """
        String representation of Dotdict object.
        
        >>> Dotdict(a=1)
        Dotdict({'a': 1})
        """
        # pprint.pformat() gives nonrandom order of items
        return "%s(%s)" % (self.__class__.__name__, pprint.pformat(self.copy()))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
