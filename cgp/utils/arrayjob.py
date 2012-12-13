"""
Simple wrapper for PBS array jobs. See test_arrayjob.py for a working example.

Usage:

#. Import the arrayjob module, usually with ``from utils.arrayjob import *``
#. Set the number of tasks that are going to split the work.
#. Define one function per stage of computation.
#. Call ``arun(stage0, par(stage1), ...)`` where ``par()`` marks stages to run 
   in parallel.

The arrayjob module senses which mode it is running in, by examining the 
environment variables PBS_ARRAYID and STAGE_ID.

* Regular execution: arun() will submit one job per stage.
* Serial job: single instance will execute stage STAGE_ID.
* Parallel job: parallel instances will execute stage STAGE_ID.

If STAGE_ID is not set, the script is executing for the first time, and 
arun() will submit a batch job for each stage, with dependencies between stages.
Parallel stages use the PBS "array job" facility. Also, Stallo won't put 
multiple jobs on the same node. We work around this by using MPI without 
actually passing any messages, just -lnodes=1:ppn=8 and running with mpirun.

If PBS_ARRAYID exists, the variable ID is set to ``PBS_ARRAYID * size + rank``, 
where size and rank are taken from MPI (if OMPI_COMM_WORLD_RANK exists), 
otherwise size, rank = 1, 0.

Example:

>>> from cgp.utils.arrayjob import *
>>> set_NID(16)                                                 # doctest: +SKIP

Next, define one function per stage of computation.
Finally, specify the sequence of stages, and which of them should run in 
parallel (as an "array job").

>>> arun(stage0, par(stage1), ...)                              # doctest: +SKIP

Importing ``* from arrayjob`` defines the following:

* :func:`arun` : Run a multi-stage job, 
  submitting parallel stages as array jobs.
* :func:`presub` : Indicate that first stage should run on login node 
  rather than batch.
* :func:`par` : Indicate that a stage should execute in parallel.
* :data:`ID` : Array job index ID (in the sequence 0, 1, ..., NID-1).
* :func:`get_NID` : Get the number of array jobs to submit, 
  or that have been submitted.
* :func:`set_NID` : Set number of array jobs to submit. 
  You should call set_NID() exactly once.
* :func:`qopt` : Decorator to pass job parameters for an individual stage
* :data:`alog` : Logger object for the arrayjob module.
* :func:`wait` : Do-nothing stage used to separate parallel stages if required.
* :func:`memmap_chunk` : Read-write memmap to chunk ID out of NID.

Calling ``set_NID(n+1)`` is equivalent to::

    qsub -t 0-n <jobscript>

or the jobscript directive ::

    #PBS -t 0-n <jobscript>

(The :func:`reset_NID` below prepares for other doctests by forgetting 
that we have called :func:`set_NID`.)

>>> reset_NID()
"""
# pylint: disable=W0621,W0603

import sys # sys.argv[0] to get name of jobscript
import os # file and directory manipulation, and PBS_O_WORKDIR
from os import environ as e
from cgp.utils.commands import getstatusoutput # calling qsub
import logging # diagnostics
from collections import defaultdict

import numpy as np

from cgp.utils.ordereddict import OrderedDict
from cgp.utils.rec2dict import dict2rec
from cgp.utils.dotdict import Dotdict

__all__ = """arun presub par ID get_NID set_NID reset_NID
             qopt alog wait memmap_chunk Mmapdict Timing""".split()

# Array job index ID sequence: 0, 1, ..., NID-1
ID = NID = ppn = rank = size = STAGE_ID = None
if "STAGE_ID" in e: # executing as batch job
    STAGE_ID = int(e["STAGE_ID"])
if "PBS_ARRAYID" in e: # executing as parallel batch job
    AID = int(e["PBS_ARRAYID"])
    if "OMPI_COMM_WORLD_RANK" in e: # running under MPI
        from mpi4py import MPI  #@UnresolvedImport
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        size, rank = 1, 0
    ID = int(AID) * size + rank

# Initialize logging. Keys refer to the dict made available by the logger.
keys = "asctime levelname name lineno process message"
fmt = "%(" + ")s\t%(".join(keys.split()) + ")s"
logging.basicConfig(level=logging.INFO, format=fmt)
alog = logging.getLogger('arrayjob')

# Path to Python jobscript (the one initially invoked)
jobscript = os.path.realpath(sys.argv[0])

class QsubException(Exception):
    """Exception in queue submission."""
    def __init__(self, status, output, cmd):
        super(QsubException, self).__init__()
        self.status = status
        self.output = output
        self.cmd = cmd
    def __str__(self):
        classname = self.__class__.__name__
        return ("%s%s" % (classname, (self.status, self.output, self.cmd)))

def set_NID(i, n=8):
    """
    Set number of array jobs to submit. You should call set_NID() exactly once.
    
    >>> arun()
    Traceback (most recent call last):
    AssertionError: Need to call set_NID() before arun()
    
    >>> set_NID(1)
    Traceback (most recent call last):
    AssertionError: NID (1) must be an even multiple of # processors per node (8)
    
    >>> set_NID(8)
    
    >>> set_NID(16)
    Traceback (most recent call last):
    AssertionError: Attempting to call set_NID() more than once
    
    >>> reset_NID() # undo effects of the doctest
    """
    global NID, ppn
    ppn = n
    assert NID is None, "Attempting to call set_NID() more than once"
    msg = "NID (%s) must be an even multiple of # processors per node (%s)"
    assert i % ppn == 0, msg % (i, ppn)
    NID = i

def reset_NID():
    """
    Reset NID to None, so ``assert NID is None`` does not fail when debugging.
    """
    global NID
    NID = None

def get_NID():
    """Get the number of array jobs to submit, or that have been submitted."""
    return NID

# Implementation note: Originally we added attributes "par" etc. to functions, 
# but this does not work with instance methods. The current approach keeps a 
# module-level dict of options. Each decorator merely records the relevant 
# option and returns the callable unchanged.

class Qopt(defaultdict):
    """Dict to record queue options."""
    @staticmethod
    def key(func):
        """Return func, or its wrapped function if decorated."""
        return func.__func__ if hasattr(func, "__func__") else func
    def __getitem__(self, func):
        return super(Qopt, self).__getitem__(key(func))
        # return self[key(func)]
    def __setitem__(self, func, value):
        super(Qopt, self).__setitem__(key(func), value)
        # self[key(func)] = value

opt = dict(par=set(), presub=set(), qopt=Qopt(list))

def par(func):
    """
    Indicate that a stage should execute in parallel.
    
    Wrapping a function in :func:`par` adds it to the set of functions that the 
    arrayjob module knows should be executed as array jobs.
    
    >>> def f(): pass
    >>> is_par(par(f))
    True
    
    It can be used as a decorator.
    
    >>> @par
    ... def h(): pass
    >>> is_par(h)
    True
    """
    opt["par"].add(key(func))
    return func

def is_par(func):
    """
    Return ``True`` if a function has been marked by par(stage).
    
    >>> def f(): pass
    >>> is_par(f)
    False
    >>> is_par(par(f))
    True
    """
    return key(func) in opt["par"]

def presub(func):
    """
    Indicate that first stage should run on login node rather than as batch job.
    
    Cheap initialization can run on the login node without waiting in queue, 
    saving some time.
    
    Typical usage, assuming setup(), work() and wrapup() have been defined.
    
    >>> arun(presub(setup), par(work), wrapup)                  # doctest: +SKIP
    
    >>> def f(): pass
    >>> is_presub(presub(f))
    True
    
    It can be used as a decorator.
    
    >>> @presub
    ... def h(): pass
    >>> is_presub(h)
    True
    
    Instance methods are OK, whether bound or unbound.
    
    >>> class A(object):
    ...     @presub
    ...     def test(self):
    ...         pass
    >>> a = A()
    >>> is_presub(a.test)
    True
    >>> is_presub(A.test)
    True
    
    This example shows that presub(f) is executed immediately by arun.
    (Here, no jobs are submitted because there are no regular stages.)
    
    >>> set_NID(8)
    >>> def f():
    ...     print "Testing"
    >>> arun(presub(f))
    Testing
    >>> reset_NID() # undo effects of the doctest
    """
    opt["presub"].add(key(func))
    return func

def is_presub(func):
    """
    Return True if a function has been marked by presub(stage).
    
    >>> def f(): pass
    >>> is_presub(f)
    False
    >>> is_presub(presub(f))
    True
    """
    return key(func) in opt["presub"]

def key(func):
    """
    Common basis for comparison of functions, bound and unbound methods.
    
    >>> class A(object):
    ...     def test():
    ...         pass
    >>> a = A()
    >>> a.test, A.test
    (<bound method A.test of <...A object at 0x...>>, 
     <unbound method A.test>)
    >>> key(a.test), key(A.test)
    (<function test at 0x...>, <function test at 0x...>)
    >>> a.test == A.test
    False
    >>> key(a.test) == key(A.test)
    True
    """
    return func.__func__ if hasattr(func, "__func__") else func

def qopt(*args):
    """
    Decorator to associate queueing options with a stage function.
    
    >>> @qopt("-l walltime=00:01:00", "-j oe")
    ... def f():
    ...     pass
    
    The options are stored internally in a list.
    
    >>> opt["qopt"][f]
    ['-l walltime=00:01:00', '-j oe']
    
    This works with any callable, including instance methods.
    
    >>> class A(object):
    ...     @qopt("-j oe")
    ...     def test(self):
    ...         pass
    >>> a = A()
    >>> opt["qopt"][key(a.test)]
    ['-j oe']
    
    Can be applied multiple times. The outermost option is added last.
    
    >>> @qopt("-j oe")
    ... @qopt("-l walltime=00:01:00")
    ... def g():
    ...     pass
    >>> opt["qopt"][g]
    ['-l walltime=00:01:00', '-j oe']
    """
    def wrapper(func):  # pylint:disable=C0111
        opt["qopt"][key(func)].extend(args)
        return func
    return wrapper

def wait():
    """Do nothing. Used to separate parallel stages if required."""
    pass

def array_opt(NID):
    """
    PBS array job option.
    
    >>> array_opt(3)
    '-t 0-2'
    """
    return "-t 0-%s" % (NID - 1)

def submit(STAGE_ID, *options):
    """
    Submit the jobscript as an array job, passing any *options to qsub.
    
    This is equivalent to running::
    
        qsub -v STAGE_ID=2 -t 0-2 -W depend=afterok:123 arrayjob.py
    
    >>> submit(2, array_opt(3), "-W depend=afterok:123")        # doctest: +SKIP
    
    submit() returns the job ID assigned by the queue system.
    """
    if any([opt.startswith("-t") for opt in options]):
        wrapper = "qsubwrapmpi"
    else:
        wrapper = "qsubwrap"
    options = " ".join(options)
    cmd = "%s -v STAGE_ID=%s %s %s" % (wrapper, STAGE_ID, options, jobscript)
    status, output = getstatusoutput(cmd)
    result = output
    alog.info("Submitted (status, output, cmd): %s", (status, output, cmd))
    if status != 0:
        raise QsubException(status, output, cmd)
    return result.split(".", 1)[0]

def arun(*stages, **kwargs):
    """
    Run a multi-stage job, submitting parallel stages as array jobs.
    
    * ``*stages`` is a variable number of functions to execute.
    * ``**kwargs`` may contain optional keyword arguments: 'loglevel' and 
      'testID', see below.        
    
    Typical usage, assuming setup(), work() and wrapup() have been defined.
    
    >>> arun(setup, par(work), wrapup)                          # doctest: +SKIP
    
    Note that the stage functions are not called, just passed as arguments.
    Stages to execute in parallel are wrapped in par().
    The names "setup" etc. are just examples; you can have any number of stages, 
    each of which may be serial (default) or parallel (indicated by par()).
    
    :func:`arun` executes differently depending on whether the script is 
    executing normally or as a job in the batch system.
     
    When the script executes normally, arun() submits batch jobs of itself, 
    once per stage, passing a STAGE_ID environment variable to each job that 
    indicates which stage to execute.
    
    If STAGE_ID is defined, arun() knows that it is executing as a batch job 
    and simply calls the STAGE_ID'th function in ``*stages``. 
    
    Dependencies are used to link jobs together. 
    A parallel stage has a ``beforeok:`` dependency on the following stage, 
    which in turn has an ``on:NID`` dependency that waits until all array job ID's 
    have completed.
    Stages that follow a serial stage have an afterok: dependency on that stage.
    
    (Plain ``afterok:`` doesn't work with array jobs; they would have to be written 
    out as jobid-0, jobid-1, ..., and qsub won't accept very long dependency 
    options.)
    
    Two consecutive stages cannot both be parallel, because each ``on:`` must match 
    a ``beforeok:``, and neither ``beforeok:`` nor ``afterok:`` can take array job ID's.
    
    >>> set_NID(16)
    >>> def f0(): pass
    >>> def f1(): pass
    >>> arun(par(f0), par(f1))
    Traceback (most recent call last):
    AssertionError: Consecutive stages <function f0 at 0x...> and <function f1 
    at 0x...> are both parallel. qsub dependencies cannot handle this case. 
    Workaround: insert a serial arrayjob.wait.
    
    To specify a different loglevel for the logger (arrayjob.alog), pass an 
    optional keyword argument 'loglevel' whose value is the name of a loglevel 
    constant, e.g. "DEBUG" for logging.DEBUG.
    
    >>> arun(setup, par(work), wrapup, loglevel="DEBUG")        # doctest: +SKIP
    
    For testing stages without submitting jobs, pass testID=<integer>. STAGE_ID 
    will be set to 0, 1, ... in turn, with ID set to testID.
    
    >>> reset_NID() # revert side-effect of this doctest
    """
    loglevel = kwargs.pop("loglevel", "INFO")
    alog.setLevel(getattr(logging, loglevel))
    STAGE_ID = os.environ.get("STAGE_ID")
    
    def run_presub(stages):
        """Execute any pre-submission stages."""
        if is_presub(stages[0]):
            # Pop the first stage
            stage, stages = stages[0], stages[1:]
            # Execute it only if Python was invoked normally, not in a queue job
            if STAGE_ID is None:
                msg = "Only the first stage can execute on the login node"
                assert all([not is_presub(i) for i in stages]), msg
                alog.info("Pre-submit stage starting: %s", stage)
                stage()
                alog.info("Pre-submit stage done: %s", stage)
        return stages
    
    global ID  # pylint:disable=W0603
    testID = kwargs.pop("testID", None)
    if testID is not None:
        alog.info("Calling arun() with testID=%s", testID)
        stages = run_presub(stages)
        ID = testID
        for STAGE_ID in range(len(stages)):
            os.environ["STAGE_ID"] = str(STAGE_ID)
            stages[STAGE_ID]()
            # arun(*stages, loglevel=loglevel, testID=testID)
        return
    
    alog.debug("ID=%s, STAGE_ID=%s", ID, STAGE_ID)
    assert not kwargs, "Undefined keyword arguments: %s" % kwargs
    assert NID is not None, "Need to call set_NID() before arun()"
    stages = run_presub(stages)
    # Return early if there are no stages to submit jobs for
    if not stages:
        return
    if STAGE_ID is None: # not invoked as queue job, so submit jobs
        for this, next_ in zip(stages, stages[1:]):
            if is_par(this) and is_par(next_):
                msg = "Consecutive stages %s and %s are both parallel."
                msg += " qsub dependencies cannot handle this case."
                msg += " Workaround: insert a serial arrayjob.wait."
                raise AssertionError(msg % (this, next_))
        on_opt = "-W depend=on:%s" % (NID / ppn)
        jobid = {} # storing receipts from qsub
        jobdep = defaultdict(list) # accumulating dependencies
        # boolean vector: whether each stage follows a parallel one
        afterpar = np.array([(STAGE_ID > 0) and is_par(stages[STAGE_ID - 1]) 
            for STAGE_ID, stage in enumerate(stages)])
        # array of (STAGE_ID, stage) tuples
        istage = np.array(list(enumerate(stages)))
        for STAGE_ID, stage in istage[afterpar]:
            # submit each stage that follows a parallel one
            jobid[STAGE_ID] = submit(STAGE_ID, on_opt, *opt["qopt"][stage])
            # prepare for the parallel one to depend on the one just submitted
            jobdep[STAGE_ID - 1].append("beforeok:%s" % jobid[STAGE_ID])
        for STAGE_ID, stage in istage[~afterpar]:
            # submit each stage that does not follow a parallel one
            if STAGE_ID > 0:
                # prepare for this stage to depend on the previous one
                jobdep[STAGE_ID].append("afterok:%s" % jobid[STAGE_ID - 1])
            dep = jobdep[STAGE_ID]
            dep_opt = ("-W depend=" + ",".join(dep)) if dep else ""
            arr_opt = array_opt(NID / ppn) if is_par(stage) else ""
            jobid[STAGE_ID] = submit(STAGE_ID, dep_opt, arr_opt, 
                *opt["qopt"][stage])
    else: # invoked as queue job
        stage = stages[int(STAGE_ID)]
        alog.info("Stage starting: %s", stage)
        stage()
        alog.info("Stage done: %s", stage)

def memmap_chunk(filename, mode="r+", **kwargs):
    """
    Read-write memmap to chunk ID out of NID.
    
    ID and NID are normally taken from arrayjob.get_ID() and arrayjob.get_NID(), 
    but can be specified as keyword arguments for testing.
    
    Assuming a Numpy array has already been saved to file.
    
    >>> np.save("test.npy", np.arange(10))
    
    Return chunk 1 out of [0, 1, 2], [3, 4, 5], [6, 7], [8, 9].
    
    >>> c = memmap_chunk("test.npy", ID=1, NID=4)
    >>> c
    memmap([3, 4, 5])
    
    The offset attribute is the original index of the first item in the chunk.
    
    >>> c.offset
    3
    
    Clean up after the test.
    
    >>> del c
    >>> os.remove("test.npy")
    """
    myID = kwargs.get("ID", ID)
    myNID = kwargs.get("NID", NID)
    # deferred import so arrayjob can be used without load_memmap_offset
    from cgp.utils.load_memmap_offset import open_memmap, memmap_chunk_ind
    r = open_memmap(filename, "r")
    n = r.shape[0]
    i = np.array_split(range(n), myNID)[myID]
    if len(i) > 0:
        result, offset = memmap_chunk_ind(filename, i, mode=mode)
        result.offset = offset
        return result
    else:
        return np.empty(0, r.dtype)

class Mmapdict(Dotdict):
    """
    Dictionary that memory-maps an existing {key}.npy on first lookup of d[key].
    
    Attribute access ``d.key`` works too.
    
    >>> import tempfile, shutil
    >>> dtemp = tempfile.mkdtemp()
    >>> try:
    ...     np.save("%s/a.npy" % dtemp, np.arange(3))
    ...     md = Mmapdict(pardir=dtemp, mode="r")
    ...     print repr(md["a"])
    ...     print repr(md.a)
    ...     del md
    ...     # It can be useful to pass shape and offset to Mmapdict()
    ...     np.save("%s/b.npy" % dtemp, np.arange(4))
    ...     md = Mmapdict(pardir=dtemp, mode="r+", shape=(2,), offset=1)
    ...     print repr(md["b"])
    ...     md["b"][:] = 10 + np.arange(len(md["b"]))
    ...     del md
    ...     # The memory-mapped part is modified
    ...     np.load("%s/b.npy" % dtemp)
    ... finally:
    ...     shutil.rmtree(dtemp)
    memmap([0, 1, 2])
    memmap([0, 1, 2])
    memmap([1, 2])
    array([ 0, 10, 11,  3])
    """
    def __init__(self, pardir=os.curdir, **kwargs):
        """
        Mmapdict constructor. Create pardir if not exists.
        
        :param str pardir: parent directory of preexisting .npy files to be memory-mapped.
        :param: ``**kwargs`` : passed to :func:`open_memmap`.
        """
        super(Mmapdict, self).__init__()
        if not os.path.exists(pardir):
            os.makedirs(pardir)
        self.pardir = pardir
        self.kwargs = kwargs
    
    def __missing__(self, key):
        """Memory-map existing {key}.npy on first lookup of key."""
        # deferred import so arrayjob can be used without load_memmap_offset
        from cgp.utils.load_memmap_offset import open_memmap
        return open_memmap("%s/%s.npy" % (self.pardir, key), **self.kwargs)

class Timing(OrderedDict):
    """
    :class:`OrderedDict` for timings.
    
    >>> Timing(attempts=1)
    Timing([('attempts', 1), ('waiting', nan), ('started', nan), 
        ('finished', nan), ('error', nan), ('seconds', nan)])
    """
    
    _fields = "attempts waiting started finished error seconds".split()
    _default = OrderedDict((k, np.nan) for k in _fields)
    _default["attempts"] = np.int64(0)
    
    def __init__(self, **kwargs):
        """
        Constructor for :class:`Timing`. Keyword arguments override default (0, NaN, ...)
        
        Overriding defaults.
        
        >>> Timing(attempts=1)
        Timing([('attempts', 1), ('waiting', nan), ('started', nan), 
            ('finished', nan), ('error', nan), ('seconds', nan)])
        """
        super(Timing, self).__init__()
        for k, v in self._default.items():
            self[k] = v
        for k, v in kwargs.items():
            self[k] = v
    
    def __array__(self):
        """
        Convert timing to record array.
        
        >>> np.array(Timing())  # ...ellipsis to allow 0 or 0L
        array([(0..., nan, nan, nan, nan, nan)],
            dtype=[('attempts', '<i8'), ('waiting', '<f8'), ('started', '<f8'), 
            ('finished', '<f8'), ('error', '<f8'), ('seconds', '<f8')])
        """
        return dict2rec(self)
    
    def item(self):
        """
        Emulate .item() method of np.recarray.
        
        >>> Timing().item()  # ...ellipsis to allow 0 or 0L
        (0..., nan, nan, nan, nan, nan)
        """
        return np.array(self).item()

d = os.environ.get("PBS_O_WORKDIR")
if d:
    alog.info("Changing to PBS_O_WORKDIR directory: %s" % d)
    os.chdir(d)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE)
