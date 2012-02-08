"""Demo of infrastructure for large-scale cGP studies."""

import numpy as np

from cgp.examples.basic import genotypes, gt2par, par2ph, ph2agg, summarize

def hdfcaching():
    """Auto-cache/save results to HDF."""
    import os
    import tables as pt    
    from cgp.utils.hdfcache import Hdfcache
    
    filename = "/home/jonvi/hdfcache.h5"
    hdfcache = Hdfcache(filename)
    pipeline = [hdfcache.cache(i) for i in gt2par, par2ph, ph2agg]
    with hdfcache:
        for i in genotypes:
            for func in pipeline:
                i = func(i)
    with pt.openFile(filename) as f:
        gt = f.root.gt2par.input[:]
        agg = f.root.ph2agg.output[:]
    summarize(gt, agg)
    os.system("h5ls -r " + filename)

def clusterjob():
    """Splitting tasks as arrayjobs on a PBS cluster."""
    import os
    from cgp.utils import arrayjob
    arrayjob.set_NID(8)
    
    def workpiece(gt):
        """Map a single genotype to parameter to phenotype to aggregate."""
        par = gt2par(gt)
        ph = par2ph(par)
        agg = ph2agg(ph)
        return par, ph, agg
    
    def setup():
        """Generate genotypes and allocate result arrays on disk."""
        if not os.path.exists("gt.npy"):
            np.save("gt.npy", genotypes)
            # Pass the first genotype through the pipeline 
            # to get dtypes for allocating result files for memory-mapping.
            par, ph, agg = workpiece(genotypes[0])
            # Preallocate arrays on disk (can be larger than available memory).
            # Changes are written back to disk when array goes out of scope.
            d = dict(par=par, ph=ph, agg=agg)
            for k, v in d.items():
                a = np.memmap(k + ".npy", v.dtype, "w+", shape=len(genotypes))
                a[0] = v
    
    def task():
        """Process a chunk of workpieces using :func:`memmap_chunk` magic."""
        filenames = [i + ".npy" for i in "gt par ph agg".split()]
        gt, par, ph, agg = [arrayjob.memmap_chunk(i) for i in filenames]
        for i in range(len(gt)):
            par[i], ph[i], agg[i] = workpiece(gt[i])
    
    def wrapup():
        """Summarize results once all are done."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.use("agg")
        gt = np.load("gt.npy")
        agg = np.load("agg.npy")
        summarize(gt, agg)
        plt.savefig("summary.png")
    
    arrayjob.arun(arrayjob.presub(setup), arrayjob.par(task), wrapup)
