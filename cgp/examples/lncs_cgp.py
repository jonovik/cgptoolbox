#!/usr/bin/env python
"""
Reusable functions for cGP paper on LNCS model.

Max burn-in of 1000 takes about 150 s. Most converge much faster, in about 30 s.
"""

__version__ = "$Revision$"
id = "$Id$"

from collections import deque # sliding window of action potentials
from datetime import datetime
from time import time
import os
import shutil
from warnings import warn
import sys

import numpy as np
from utils.load_memmap_offset import open_memmap

import ap_cvode # Li, ap_stats_array
from current_phenotypes import current_phenotypes_array
from protocols import Clampable, listify, decayfits, mmfits
from utils.placevalue import Genotype
from utils.thinrange import thin
from utils.rec2dict import dict2rec
from utils.dotdict import Dotdict
from utils.ordereddict import OrderedDict
from utils.numpy_hdf import numpy2hdf
from utils.unstruct import unstruct
from utils.svnversion import svnversion, svninfo
from utils.failwith import nans_like
from utils.write_if_not_exists import write_if_not_exists
from utils.arrayjob import * # chdir to PBS_O_WORKDIR, where job was submitted
from utils import arrayjob
from utils.master_worker_decorator import parallel, rank, root, comm

__all__ = "Lncs_cgp Genotype FrF2 savetohdf np timeline makeplots".split()

class Lncs_cgp(object):
    """Causally cohesive genotype-phenotype simulations of LNCS model."""
    
    def __init__(self, genotype, li=None, filename="lncs_cgp.h5", relvar=0.5, 
        winwidth=10, reltol=0.001, raw_nap=1, raw_nthin=100, max_nap=1000, 
        resumeconv=False, **kwargs):
        """
        Create an Lncs_cgp object that wraps genotype, options and GP map.
        
        This class wraps "global" variables such as the model object, 
        as well as functions that rely on those global variables.
        
        If index.npy exists in the current directory, only those items will be
        processed. 
        
        genotype : Genotype object or structured array, or anything that offers 
            g[i], len(g), and g.dtype.names.
        li : LNCS model object (defaults to Clampable.mixin(Li, reltol=1e-10))
        filename : Not used yet, might pass it to save_version_info() and 
            savetohdf().
        relvar : Genotypes (aa, Aa, AA) = (0, 1, 2) give parameter values of
            (1-relvar, 1, 1+relvar) * baseline value.
        winwidth : Window width (number of intervals) in checking for 
            alternans.
        reltol : Relative tolerance criterion for convergence, applied to the 
            integral of each state variable.
        raw_nap : Number of raw action potentials to store in HDF file.
        raw_nthin : Number of time-points per interval for raw trajectories.
        max_nap : Number of action potentials before convergence check fails.
        resumeconv : Try to read y0 from steady.npy and nap from convergence.npy. 
        
        Optional keyword arguments are added as object attributes for later 
        reference.
        """
        self.resumeconv = resumeconv
        # Note that the "reltol" we pass to Li is something else than 
        # the "reltol" we use to check convergence of action potentials.
        if li:
            self.li = li
        else:
            self.li = Clampable.mixin(ap_cvode.Li, maxsteps=500000, 
                chunksize=20000, reltol=1e-10)
        if np.isscalar(reltol):
            reltol = dict((k, reltol) for k in self.li.dtype.y.names)
        self.__dict__.update(genotype=genotype, filename=filename, 
            relvar=relvar, winwidth=winwidth, reltol=reltol, raw_nap=raw_nap, 
            raw_nthin=raw_nthin, max_nap=max_nap, **kwargs)
        self.ftype = self.datatypes(genotype)
    
    def datatypes(self, genotype):
        """Dict of Numpy data types for each Table in the HDF file."""
        li = self.li # save some typing
        ftype = dict(genotype=self.genotype.dtype, parameter=li.dtype.p, 
            convergence=[("period", np.int16), ("intervals", np.int16)], 
            steady=self.li.dtype.y, quiescent=self.li.dtype.y, 
            # Hacked by stepping through self.task(0), see comment there
            clamppheno=[(k, float) for k in 
                "i_Na_tau_m40 i_Na_tau_m20 i_Na_tau_0 i_Na_tau_20 i_Na_tau_40 "
                "i_CaL_tau_m40 i_CaL_tau_m20 i_CaL_tau_0 i_CaL_tau_20 "
                "i_CaL_tau_40 i_CaL_vg_max_m90 i_CaL_vg_thalf_m90 "
                "i_CaL_vg_max_m80 i_CaL_vg_thalf_m80 i_CaL_vg_max_m70 "
                "i_CaL_vg_thalf_m70".split()],
            timing=[(k, float) for k in 
                "waiting started finished error seconds attempts".split()])
        ftype["raw"] = [(k, float, self.raw_nap * self.raw_nthin) 
            for k in ("t",) + li.dtype.y.names + li.dtype.a.names]
        
        # Ion-current phenotypes: a few phenotypes for many variables.
        # Make one array per phenotype, with one column per variable.
        
        # Get field names for current phenotypes
        with li.autorestore():
            t, y, stats = li.ap_plain()
        curph = current_phenotypes_array(t, y.V)
        for k in curph.dtype.names:
            ftype[k] = [(i, float) for i in li.dtype.y.names + li.dtype.a.names]
        ap_stats = ap_cvode.ap_stats_array(stats)
        # Action potential and calcium transient statistics, including time 
        # to recovery.
        ftype["ap_stats"] = [(k, typ, self.raw_nap) for k, typ in eval(str(ap_stats.dtype))]
        return ftype

    # STAGE FUNCTIONS
    
    @presub
    def setup(self):
        """Prepare for job submission: save version info, preallocate files."""
        self.save_version_info()
        if not os.path.exists("index.npy"):
            self.allocfiles()
        else:
            self.resume() # create index.npy from unfinished items in timing.npy
    
    def save_version_info(self):
        # Store svnversion now, it's not available when running as job
        svnver = []
        warning = []
        error = []
        info = [id, "", svninfo(__file__)]
        for fname in __file__, ap_cvode.__file__: # script and cGPtoolbox
            dir = os.path.realpath(os.path.split(fname)[0])
            ver = svnversion(dir)
            if "exported" in ver:
                error.append("Not a working copy: %s" % dir)
            if "M" in ver:
                warning.append("Uncommitted changes in %s" % dir)
            svnver.append(ver + " " + dir)
            info.append(svninfo(dir))
        if warning:
            warn("\n".join(warning))
        if error:
            raise Exception("\n".join(error))
        
        if os.path.exists("index.npy"):
            warn("\nFiles exist. Will now rerun incomplete workpieces.")
        
        # Must print to sys.stderr to bypass tee's output buffering, cf.
        # http://shallowsky.com/blog/programming/python-tee.html
        if not "PBS_O_WORKDIR" in os.environ: # not running as batch job
            print >>sys.stderr, ("Ready to submit jobs. %s genotypes, %s processors. "
                "Press ENTER to confirm, or Ctrl-C to abort." % 
                (len(self.genotype), get_NID()))
            raw_input()
        
        with open("svnversion.txt", "a") as f:
            f.write("\n" + "\n".join(svnver))
        with open("svninfo.txt", "a") as f:
            f.write("\n" + "\n".join(info))
    
    def allocfiles(self, **kwargs): # **kwargs override fname[k]
        """
        Save version info and preallocate empty memory-mapped arrays on disk.
        
        The arrays are taken from self.ftype, but may be overridden by 
        optional keyword arguments. For example, allocfiles(genotype=...) will 
        save the provided genotype as "genotype.npy" rather than allocate an 
        empty file of that name with dtype = ftype["genotype"].dtype.
        """
        genotype = kwargs.pop("genotype", self.genotype)
        shape = (len(genotype),) # must be tuple, or open_memmap creates broken file
        np.save("index.npy", np.arange(len(genotype)))
        for name, dtype in self.ftype.items():
            fname = "%s.npy" % name
            assert not os.path.exists(fname)
            if name in kwargs:
                np.save(fname, kwargs.pop(name))
            else:
                f = open_memmap(fname, mode="w+", dtype=dtype, shape=shape)
                if name == "timing":
                    f[:] = nans_like(np.empty(dtype=dtype, shape=shape))
                    f["waiting"] = time()
                    f["attempts"] = 0
                del f
        assert not kwargs, "Unrecognized keyword argument(s) %s" % kwargs.keys()

    def resume(self):
        """Prepare to rerun unfinished workpieces."""
        timing = open_memmap("timing.npy")
        indexbool = ~np.isfinite(timing["finished"])
        if self.resumeconv:
            c = np.load("convergence.npy").view(np.recarray)
            indexbool = indexbool | ((c.intervals > 0) & (c.period == 0))
        index = np.where(indexbool)[0]
        if len(index) == 0:
            raise Exception("No workpieces to resume")
        np.save("index.npy", index)
        timing[index]["waiting"] = time()
        timing[index]["attempts"] += 1
        del timing
    
    def task(self, real_i):
        """
        Map genotype i to parameter to steady state.
        
        Input and output uses memmaps. To allow load balancing, the memmaps 
        have only a single element (one genotype, one parameter set, etc.).
        
        In case of alternans, a list of (t, y, stats, integrals) is stored as e.g.
        alternans/123.list.npy
        To retrieve the list, use "L = np.load(filename).item()"
        """
        m = Dotdict()
        for name in self.ftype:
            m[name] = open_memmap("%s.npy" % name, offset=real_i, shape=1)
        i = 0
        alog.debug("Workpiece: real_i=%s, i=%s", real_i, i)
        m.timing[i]["attempts"] += 1
        m.timing[i]["started"] = time()
        try:
            m.genotype[i] = gt = self.genotype[real_i]
            m.parameter[i] = par = self.gt2par(gt)
            
            li = self.li
            
            with li.autorestore(_p=par):
                
                # Compute quiescent state
                with li.autorestore(stim_amplitude=0):
                    t, y, flag = li.integrate(t=[0, 1e6])
                m.quiescent[i] = yq = y[-1:] # yq now has shape (1,)
                
                # Run voltage stepping protocols
                bp = li.bond_protocols()
                p3, p5, p7 = [listify(bp[j].protocol) for j in 3, 5, 7]
                p3[1][1] = -40, -20, 0, 20, 40
                p5[1][1] = -40, -20, 0, 20, 40
                p7[2][0] = 2, 127, 252, 377, 502
                with li.autorestore(_y=yq):
                    L3 = li.vecvclamp(p3) # P1-P2 voltage stepping for i_Na
                    L5 = li.vecvclamp(p5) # P1-P2 voltage stepping for i_CaL
                    L7 = li.vecvclamp(p7) # Variable-gap protocol for i_CaL
                
                d = OrderedDict() # to hold voltage-clamp phenotype characteristics
                
                # Time constant of inactivation: -1/slope of ln(current) vs t after P1 peak
                for curr, L in ("i_Na", L3), ("i_CaL", L5):
                    for v, tau in zip(*decayfits(L, 1, curr)):
                        sv = str(v).replace("-", "m") # avoid minus sign in field name
                        k = "%s_tau_%s" % (curr, sv)
                        d[k] = tau
                
                # Michaelis-Menten half-saturation and asymptote of peak current vs gap duration
                gaps, voltages = p7[2]
                proto, traj = [np.array(j, dtype=object) for j in zip(*L7)]
                # Vectorized voltage clamping returns a 1-d array.
                # Dimensions should be (gap length, interpulse voltage, pulse index, last)
                # where the last dimension has length 2 for proto (duration, voltage) 
                # and 4 for traj (t, y, dy, a).
                for j in proto, traj:
                    j.shape = [len(gaps), len(voltages)] + list(j.shape[1:])
                for j, v in enumerate(voltages):
                    imax, thalf = mmfits(zip(proto[:,j,:,:], traj[:,j,:,:]), k="i_CaL")
                    sv = str(v).replace("-", "m")
                    d["i_CaL_vg_max_" + sv] = imax
                    d["i_CaL_vg_thalf_" + sv] = thalf
                # print " ".join(dict2rec(d).dtype.names) # for self.datatypes()
                m.clamppheno[i] = dict2rec(d).view(np.recarray)
                
                # Steady state under regular pacing
                (period, intervals), steady = li.par2steady(y0=yq, reltol=self.reltol, max_nap=1000)

                # # oldintervals will be nonzero if an earlier run failed to converge
                # oldperiod, oldintervals = m.convergence[i]
                # y0 = m.steady[i] if (self.resumeconv and oldintervals) else None
                # 
                # # Run to convergence; steady is a list of (t, y, stats, integrals).
                # # The cycle is aligned so that it starts with the highest Cai peak.
                # (period, intervals), steady = self.par2steady(par, y0)
                # intervals += oldintervals
                
                m.convergence[i] = period, intervals
                if period and (len(steady) > 1):
                    with write_if_not_exists("alternans/%s.list.npy" % real_i, 
                        raise_if_exists=True) as f:
                        L = []
                        for t, y, stats, int_ in steady:
                            ix = thin(len(t), self.raw_nthin)
                            L.append((t[ix], y[ix], stats, int_))
                        np.save(f, L)
                t, y, stats, _ = steady[-1]
                m.steady[i] = y[0]
                
                # Phenotypes for steady-state dynamics
                aps = [(t, y, stats) for t, y, stats, int_ in steady[-self.raw_nap:]]
                m.raw[i], m.ap_stats[i] = self.thin_aps(aps, self.raw_nthin, par)
                # Ion-current phenotypes, also applied to other state variables.
                # One table/ndarray per phenotype, one column per variable.
                raw = m.raw[i]
                currstats = np.concatenate([current_phenotypes_array(raw["t"], raw[k]) 
                    for k in raw.dtype.names if k != "t"])
                for k in currstats.dtype.names:
                    m[k][i] = np.ascontiguousarray(currstats[k])
                m.timing[i]["finished"] = time()
                m.timing[i]["seconds"] = m.timing[i]["finished"] - m.timing[i]["started"]
        except Exception, exc:
            m.timing[i]["error"] = time()
            alog.exception("Error processing item %s", i)
    
    @par
    def work(self):
        """Load-balanced computation of workpieces, calling task()."""
        ptask = parallel(self.task) # master-worker load balancing
        # distribute work for eight processors
        if rank == root:
            index = memmap_chunk("index.npy", ID=arrayjob.ID//8, NID=get_NID()//8)
        else:
            index = []
        index = comm.bcast(index)
        for result in ptask(index):
            pass # input, work, and output are all done by ptask()
    
    def gt2par(self, genotype):
        """Genotype to parameter mapping."""
        li = self.li
        with li.autorestore():
            for k in genotype.dtype.names:
                li.pr[k] += li.pr[k] * self.relvar * (genotype[k] - 1)
            return li.pr.copy()
    
    def thin_aps(self, aps, nthin, par):
        """Thin action potentials to "nthin" points."""
        t = np.concatenate([ti[thin(len(ti), nthin)] for ti, yi, si in aps])
        t = t - t[0]
        y = np.concatenate([yi[thin(len(ti), nthin)] for ti, yi, si in aps])
        rates, a = self.li.rates_and_algebraic(t, y, par)
        ap_stats = np.zeros(1, dtype=self.ftype["ap_stats"])
        stats = np.concatenate([ap_cvode.ap_stats_array(si) for ti, yi, si in aps])
        for k in stats.dtype.names:
            ap_stats[k] = stats[k].squeeze()
        raw = np.zeros(1, dtype=self.ftype["raw"])
        raw["t"] = t
        for k in y.dtype.names:
            raw[k] = y[k].squeeze()
        for k in a.dtype.names:
            raw[k] = a[k].squeeze()
        return raw, ap_stats

    def par2steady(self, par, y0=None):
        """
        Run heart cell to approximate steady state.
        
        Return value is ((period, number of intervals to convergence), steady) where 
        steady is a list of (t, y, stats, int_) for the last "period" intervals.
        int_ contains the integral of each state variable's trajectory. 
        The cycle is aligned so that the highest Cai peak occurs in steady[0].
        
        If dynamics does not converge within "max_nap" intervals, period is zero.
        """        
        return self.li.par2steady(par, y0, 
            winwidth=self.winwidth, max_nap=self.max_nap, reltol=self.reltol)

@qopt("-lpmem=16000MB", "-lwalltime=2:00:00")
def savetohdf(filename="lncs_cgp.h5"):
    """Create HDF5 file with one table per .npy file in current directory."""
    fileroot, ext = os.path.splitext(filename)
    rawname = os.path.join(fileroot + "_raw" + ext)
    # Move any existing HDF files to backup (will fail if backup already exists)
    if any([os.path.exists(i) for i in filename, rawname, "raw"]):
        backupdir = datetime.now().strftime("backup%Y%m%dT%H%M%S")
        os.makedirs(backupdir)
        for fname in filename, rawname:
            if os.path.exists(fname):
                shutil.move(fname, os.path.join(backupdir, fname))
    # Hack to keep raw trajectories in separate file
    os.makedirs("raw")
    shutil.move("raw.npy", "raw/raw.npy")
    # index.npy might not include all records if we resumed an earlier run
    np.save("index.npy", np.arange(len(np.load("genotype.npy"))))
    numpy2hdf(os.curdir, filename, recursive=False)
    numpy2hdf("raw", rawname)
    shutil.move("raw/raw.npy", "raw.npy")
    os.rmdir("raw")
    # Annotate hdf file
    import tables as pt
    for fname in filename, rawname:
        with pt.openFile(fname, "a") as f:
            for k in "svnversion", "svninfo", "submit":
                with open("%s.txt" % k) as g:
                    f.root._v_attrs[k] = g.read()
    # Human-readable timings
    with pt.openFile(filename, "a") as f:
        timing = f.root.timing[:]
        timstr = np.zeros(shape=timing.shape, dtype=np.dtype([(k, "S32") 
            for k in timing.dtype.names if k not in ["seconds", "attempts"]]))
        for i, t in enumerate(timing):
            timstr[i] = tuple([
                ("" if np.isnan(t[k]) else str(datetime.fromtimestamp(t[k]))) 
                for k in timstr.dtype.names])
        f.createTable(f.root, "timstr", timstr)

def FrF2(genenames, runs):
    """
    Genotypes: Fractional factorial design for two levels of a variable.
    
    @todo: There is a conflict between pylab plotting and the use of tcltk in R.
    Try to avoid plotting after running genotypes(). Plotting in an existing 
    figure window seems to work, though.
    """
    from rnumpy import r, rcopy
    r["genenames"] = rcopy(genenames)
    r("""
    library(FrF2)
    factor.names <- rep(list(c(0,2)), length(genenames))
    names(factor.names) <- genenames
    des <- FrF2(%s, factor.names=factor.names, randomize=FALSE)
    mat <- data.matrix(des)
    """ % runs)
    # r.mat has values 1 and 2 as seen from Python, even though we defined 
    # them as factor levels 0 and 2 in R.
    genotype = np.rec.fromarrays((r.mat.T - 1) * 2, names=genenames)
    return genotype

# Summarizing

def timeline(timing=None, infile="timing.npy", outfile="timing.png", dpi=100, 
    inset_n=0.05, inset_lbwh=(0.2, 0.4, 0.3, 0.5)):
    """
    Draw timeline based on recarray timing with fields started, finished.
    
    This takes about 60 s for 60000 records. Example:
    
    python -c "from lncs_cgp import timeline; timeline()"
    """
    import matplotlib as mpl
    mpl.use("agg")
    from matplotlib.pyplot import gca, gcf, axis, axes, xticks, setp
    from matplotlib.pyplot import xlabel, ylabel, plot, savefig, title
    from matplotlib.collections import LineCollection
    if timing is None:
        timing = np.load(infile)
    timing = timing.view(np.recarray)
    ax = gca()
    id = np.arange(len(timing))
    t0 = timing.started.min() # or timing.waiting.min()
    lc = LineCollection([((s - t0, i), (f - t0, i)) 
        for i, s, f in zip(id, timing.started, timing.finished)])
    setp(lc, lw=0.1, antialiased=False)
    ax.add_collection(lc)
    s = ax.scatter(timing.finished, id, np.isfinite(timing.error))
    axis([0, max(timing.finished.max(), timing.error.max()) - t0, 
          0, len(timing)])
    gcf().set_size_inches(6, 20)
    xlabel("Seconds", fontsize="small")
    ylabel("Workpiece ID", fontsize="small")
    if not dpi:
        bb = gca().get_window_extent()
        dpi = 3 * len(lc._paths) * gcf().dpi / bb.height
    if outfile:
        title(os.path.realpath(outfile), fontsize="xx-small")
    if inset_n:
        if np.array(inset_n).dtype.kind == "f":
            inset_n = inset_n * len(lc._paths)
        x0 = timing.started[:inset_n].min() - t0
        x1 = timing.finished[:inset_n].max() - t0
        box = LineCollection([zip((x0, x1, x1, x0), (0, 0, inset_n, inset_n))])
        ax.add_collection(box)
        insax = gcf().add_axes(inset_lbwh)
        axes(insax)
        timeline(timing=timing[:inset_n], outfile=None, inset_n=None)
        xlabel("")
        ylabel("")
        locs, labels = xticks()
        setp(labels, rotation=90)
    if outfile:
        savefig(outfile, dpi=dpi)
    return lc, s, dpi

@qopt("-lpmem=16000MB")
def makeplots():
    try:
        timeline()
    except Exception, exc:
        alog.exception(exc)
    import fig_raw_ap_cat
    try:
        fig_raw_ap_cat.fig()
    except Exception, exc:
        alog.exception(exc)
    fig_raw_ap_cat.tmax = None
    try:
        fig_raw_ap_cat.fig(figname="fig_raw_ap_cat_full.png")
    except Exception, exc:
        alog.exception(exc)
    import fig_splom_inset
    try:
        fig_splom_inset.main()
    except Exception, exc:
        alog.exception(exc)
    try:
        from commands import getstatusoutput
        status, output = getstatusoutput("Rscript ~/svn/jov/cgp-devel/lncs_cgp.R")
        assert status == 0, output
    except Exception, exc:
        alog.exception(exc)

# IMPLEMENTATION PRINCIPLES
# Dataflow between stages is by Numpy arrays.
# Each stage reads input from file and writes output to file.
# Using memory-mapped Numpy arrays makes this similar to Unix piping.

# NESTING HDF TABLES
# Note that pytables cannot handle a field that 
# ...contains a vector whose items have fields ("a", "b")
# but can handle one that
# ...has fields ("a", "b") whose items are vectors
# http://article.gmane.org/gmane.comp.python.pytables.user/1499
