"""
Using joblib to cache and parallelize the computations in simple.py.

Caching can be conveniently applied using decorator syntax:

.. code-block:: none
   
   @mem.cache
   def f(x):
       ...

However, parallelization won't work with this syntax as of joblib 0.6.3.
"""
# pylint: disable=R0913, R0914, W0142, C0111

import os
from tempfile import gettempdir
from multiprocessing import cpu_count

from joblib import Parallel, delayed, Memory

from cgp.examples.simple import *  # @UnusedWildImport pylint: disable=W0614

if __name__ == "__main__":
    
    cachedir = os.path.join(gettempdir(), "simple_joblib")
    mem = Memory(cachedir)
    parallel = Parallel(n_jobs=cpu_count())
    _ph = delayed(mem.cache(ph))
    
    result = np.concatenate(parallel(_ph(gt) for gt in gts)).view(np.recarray)
    
    visualize(result)
