"""
Using IPython.parallel to parallelize the computations in simple.py.

``ipcluster start`` must be run for this example to work.

Parallelization can be conveniently applied using decorator syntax:

.. code-block:: none
   
   @lv.parallel()
   def f(x):
       ...

However, cgp.examples.simple.ph is left undecorated so we can illustrate 
different tools for caching and parallelization. 
"""
# pylint: disable=W0621,E0102

from IPython.parallel import Client

from cgp.examples.simple import *  # @UnusedWildImport pylint: disable=W0614

if __name__ == "__main__":
    c = Client()
    lv = c.load_balanced_view()
    
    @lv.parallel(block=True)
    def ph(gt):
        """Import and computation to be run on engines."""
        from cgp.examples.simple import ph  # @Reimport
        rec = ph(gt)
        # Unfortunately, record arrays are not serialized correctly by IPython,
        # so return a simpler view and the correct dtype separately.
        return rec.view(float), rec.dtype
    
    result = [rec.view(dtype) for rec, dtype in ph.map(gts)]
    result = np.concatenate(result).view(np.recarray)
    
    visualize(result)
