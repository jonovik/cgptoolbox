"""
Cache computations to HDF file for simple genotype-phenotype example.

Here, the line "ph = hdfcache.cache(ph)" is equivalent to the decorator syntax:

.. code-block:: none
   
   @hdfcache.cache
   def ph(...):
       ...

However, cgp.examples.simple.ph is left undecorated so we can illustrate 
different tools for caching and parallelization. 
"""

import os
from tempfile import gettempdir

from cgp.utils.hdfcache import Hdfcache
from cgp.examples.simple import *  # @UnusedWildImport pylint: disable=W0614

if __name__ == "__main__":
    
    hdfcache = Hdfcache(os.path.join(gettempdir(), "cgpdemo.h5"))
    ph = hdfcache.cache(ph)
    
    result = np.concatenate([ph(gt) for gt in gts]).view(np.recarray)
    
    visualize(result)
