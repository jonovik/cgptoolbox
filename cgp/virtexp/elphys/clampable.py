"""Virtual voltage clamping."""
# pylint: disable=C0111,C0302

# Integer division may give zero axis range, causing LinAlgError on display
from __future__ import division

import logging
from collections import namedtuple
from contextlib import contextmanager
from itertools import chain

import numpy as np

# TODO: We now depend on rpy2, so don't need the workaround below.
try:
    from cgp.utils.rnumpy import r, RRuntimeError
    have_rnumpy = True
except ImportError:
    have_rnumpy = False
    
    import warnings
    warnings.warn("rnumpy not installed, some functions will not work.")
    
    from nose.plugins.skip import SkipTest
    
    class DummyR(object):
        """Dummy R object to skip nosetests if R is unavailable."""
        
        def __getattr__(self, name):
            raise SkipTest("rnumpy not installed")
    
    r = DummyR()

from ...cvodeint.namedcvodeint import Namedcvodeint
from . import paceable
from .ap_stats import apd
from ...utils.ordereddict import OrderedDict
from ...utils.thinrange import thin
from ...utils.splom import r2rec

Pace = namedtuple("Pace", "t y dy a stats")
Trajectory = namedtuple("Trajectory", "t y dy a")
Bond_protocol = namedtuple("Bond_protocol", "varnames protocol limits url")

# Initialize logging. Keys refer to the dict made available by the logger.
keys = "asctime levelname name lineno process message"
format_ = "%(" + ")s\t%(".join(keys.split()) + ")s"
logging.basicConfig(level=logging.INFO, format=format_)
logger = logging.getLogger("protocols")

@contextmanager
def roptions(**kwargs):
    """
    Temporarily change R options.
    
    Keyword arguments are passed to the options() function in R, restoring the 
    original settings on exiting the with block.
    
    Example: Temporarily use a different character for the decimal point.
    
    >>> with roptions(OutDec="@"):
    ...     r.as_character(1.5)[0]
    '1@5'
    
    Verify that the default has been restored.
    
    >>> r.as_character(1.5)[0]
    '1.5'

    Underscores in keyword names are automatically converted to dots, 
    so the Python statement::
    
        with roptions(show_error_messages=False):
            ...
    
    corresponds to the R statement::
    
        opt <- options(show.error.messages=FALSE)
        ...
        options(opt)
    
    One hack is that *deparse_max_lines* = 0 will suppress R tracebacks 
    altogether,     whereas in R this setting is ignored if it is not a 
    positive integer.
    """
    opt = r.options(**kwargs)
    old_max_lines = RRuntimeError.max_lines
    if kwargs.get("deparse_max_lines") == 0:
        RRuntimeError.max_lines = 0
    try:
        yield
    finally:
        r.options(opt)
        RRuntimeError.max_lines = old_max_lines

def listify(sequence):
    """
    Make a mutable copy of a sequence, recursively converting to list.
    
    Strings are left alone.
    
    This is useful for modifying pacing protocols.
    
    >>> from cgp.virtexp.elphys.examples import Bond
    >>> b = Bond()
    >>> listify(b.bond_protocols(thold=5000)[11])
    [['i_Kr'], [[5000, -80], [1000, [-70, -60, ..., 60]], [1000, -40]], 
    [[0, 2000, 0, 1]], 'http://...']
    
    >>> listify([(1, 2), (3, 4, 5), (6, [7, 8], "abc")])
    [[1, 2], [3, 4, 5], [6, [7, 8], 'abc']]
    """    
    if isinstance(sequence, basestring):
        return sequence
    else:
        try:
            return [listify(i) for i in sequence]
        except TypeError: # not iterable
            return sequence

def pairbcast(*pairs):
    """
    Broadcasting for pairs of values.
    
    No broadcasting if all pairs have scalar items.
    
    >>> pairbcast((1, 2), (3, 4))
    [[(1, 2), (3, 4)]]
    
    First item of first pair is a 2-tuple.
    
    >>> pairbcast(((1, 2), 3), (4, 5))
    [[(1, 3), (4, 5)], [(2, 3), (4, 5)]]
    
    Second item of second pair is a 2-tuple.
    
    >>> pairbcast((1, 2), (3, (4, 5)))
    [[(1, 2), (3, 4)], [(1, 2), (3, 5)]]
    
    Both items of first pair are tuples.
    
    >>> pairbcast(((1, 2), (3, 4)), (5, 6))
    [[(1, 3), (5, 6)], [(1, 4), (5, 6)], [(2, 3), (5, 6)], [(2, 4), (5, 6)]]
    
    Both items of first pair, and second item of last pair, are tuples.
    
    >>> pairbcast(((1, 2), (3, 4)), (5, 6), (7, (8, 9)))
    [[(1, 3), (5, 6), (7, 8)], [(1, 3), (5, 6), (7, 9)], 
     [(1, 4), (5, 6), (7, 8)], [(1, 4), (5, 6), (7, 9)], 
     [(2, 3), (5, 6), (7, 8)], [(2, 3), (5, 6), (7, 9)], 
     [(2, 4), (5, 6), (7, 8)], [(2, 4), (5, 6), (7, 9)]]
    """
    # flat = d0 v0 d1 v1 ..., where each item can be scalar or vector
    flat = np.array([i for p in pairs for i in p], dtype=object)
    
    # Will np.broadcast_arrays() only the vector-valued items of `flat`
    need_bc = np.array([len(np.atleast_1d(i)) > 1 for i in flat], dtype=bool)
    if not any(need_bc):
        # Standardize to a sequence (here, of length 1) of sequences of pairs
        return [[tuple(pair) for pair in pairs]]
    else:
        # bc = Cartesian product of the vector-valued items of `flat`
        bc = np.broadcast_arrays(*np.ix_(*flat[need_bc]))
        # Element number "i" of bc is a vector along the i-th dimension.
        # The limit of np.ndarray.ndim <= 32 means that we can have at most 32 
        # vector-valued items of `flat`.
        
        # Zip up corresponding items of the broadcasted, flattened vectors
        flatbc = zip(*(np.ravel(i) for i in bc))
        
        result = []
        for bc in flatbc:
            # Place the broadcasted items among copies of the scalar ones
            item = np.copy(flat)
            item[need_bc] = bc
            # Zip back into pairs
            result.append(zip(item[::2], item[1::2]))
        return result

def ndbcast(*tuples):
    """
    Broadcasting for tuples of values.

    No broadcasting if all tuples have scalar items.
    
    >>> ndbcast((1, 2, 3), (4, 5, 6))
    [[(1, 2, 3), (4, 5, 6)]]
    
    First item of first tuple is a 2-tuple.
    
    >>> ndbcast(((1, 2), 3, 4), (5, 6, 7))
    [[(1, 3, 4), (5, 6, 7)], [(2, 3, 4), (5, 6, 7)]]
    
    Second item of second tuple is a 2-tuple.
    
    >>> ndbcast((1, 2, 3), (4, (5, 6), 7))
    [[(1, 2, 3), (4, 5, 7)], [(1, 2, 3), (4, 6, 7)]]
    
    All items of first tuple are tuples.
    
    >>> ndbcast(((1, 2), (3, 4), (5, 6)), (7, 8, 9))
    [[(1, 3, 5), (7, 8, 9)], [(1, 3, 6), (7, 8, 9)], 
     [(1, 4, 5), (7, 8, 9)], [(1, 4, 6), (7, 8, 9)], 
     [(2, 3, 5), (7, 8, 9)], [(2, 3, 6), (7, 8, 9)], 
     [(2, 4, 5), (7, 8, 9)], [(2, 4, 6), (7, 8, 9)]]
    
    :func:`ndbcast` is a generalization of :func:`pairbcast`.
    
    >>> pairs = ((1, 2), (3, 4)), (5, 6), (7, (8, 9))
    >>> ndbcast(*pairs) == pairbcast(*pairs)
    True
    """
    n = len(tuples[0])
    assert all(len(p) == n for p in tuples)
    # a00 a01 a02 ... a0n a10 a11 a12 ... a1n ... amn
    flat = [np.atleast_1d(i) for p in tuples for i in p]
    pb = np.broadcast_arrays(*np.ix_(*flat)) # Cartesian product of a00 a01 ...
    # Zip up corresponding items of the broadcasted, flattened a00 a01 ...
    flatb = zip(*(np.ravel(i) for i in pb))
    # Unflatten into tuples within each item of the Cartesian product
    return [zip(*[i[j::n] for j in range(n)]) for i in flatb]

def catrec(*args, **kwargs):
    """
    Zip and concatenate arrays, globalizing time and preserving recordness.
    
    This is a quick hack, use at your own risk.
    
    Positional arguments: tuples whose items are arrays.
    Optional keyword *globalize_time* (default: True) means to assume that the 
    first item of each tuple is "local time", and convert it to "global time" 
    in the output.
    
    >>> t = np.arange(3)
    >>> a = np.rec.fromarrays([range(6)], names="i")
    >>> tc, ac = catrec((t, a[:3]), (t, a[-3:]))
    >>> tc
    array([0, 1, 2, 2, 3, 4])
    >>> all(ac == a)
    True
    
    Without globalizing time.
    
    >>> tc, ac = catrec((t, a[:3]), (t, a[-3:]), globalize_time=False)
    >>> tc
    array([0, 1, 2, 0, 1, 2])
    """
    globalize_time = kwargs.pop("globalize_time", True)
    assert not kwargs, "Unexpected keyword argument(s): %s" % kwargs
    C = []
    for i, v in enumerate(zip(*args)):
        try:
            if (i == 0) and globalize_time:
                c = np.concatenate(paceable.globaltime(v))
            else:
                c = np.concatenate([np.atleast_1d(i) for i in v], axis=0)
            C.append(c.view(v[0].dtype, type(v[0])))
        except Exception, _exc:  # pylint: disable=W0703,W0612
            C.append(v)
    return C

def vclamp2arr(L, nthin=100):
    """
    Record array thinned from results of vectorized voltage-clamp experiment.
    
    :param list L: list of Trajectory namedtuples of (t, y, dy, a)
    :param int nthin: number of time-points to retain
    :return recarray arr: Record array with fields for 
        time, state variables, algebraic variables.
        Fields are 2-d with dimensions (protocol #, time #).
        Shape is set to (1,) rather than () for convenience; 
        record arrays of shape () do not include their dtype in their string 
        representation.
    
    Example, thinning to 3 time-points and rounding results to nearest 1/4 
    for briefer printing:
    
    >>> from cgp.virtexp.elphys.examples import Fitz
    >>> b = Fitz()
    >>> protocol = (1000, -140), (500, np.linspace(-80, 40, 4)), (180, -20)
    >>> L = zip(*[traj for proto, traj in b.vecvclamp(protocol)])
    >>> holding, p1, p2 = [vclamp2arr(i, 3) for i in L]
    >>> f = p1.view(float)
    >>> f[:] = np.round(f * 4) / 4 + 0  # adding zero to avoid -0.0
    
    The resulting record array has four fields (t, V, w, I), each with shape 
    (4, 3) for 4 protocols (P1 voltage of -80, -40, 0, 40) and 3 time-points.
    
    >>> p1
    rec.array([ ([[0.0, 117.5, 500.0], [0.0, 135.5, 500.0], 
                  [0.0, 176.0, 500.0], [0.0, 140.75, 500.0]], 
                 [[-80.0, -80.0, -80.0], [-40.0, -40.0, -40.0], 
                  [0.0, 0.0, 0.0], [40.0, 40.0, 40.0]], 
                 [[-46.75, -30.0, -26.75], [-46.75, -17.75, -13.25], 
                  [-46.75, -3.25, 0.0], [-46.75, 6.0, 13.25]], 
                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 
                  [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])], 
    dtype=[('t', '<f8', (4, 3)), ('V', '<f8', (4, 3)), 
           ('w', '<f8', (4, 3)), ('I', '<f8', (4, 3))])
    """
    first = L[0]
    shape = len(L), nthin
    dtype = [(k, float, shape) 
        for k in chain(["t"], first.y.dtype.names, first.a.dtype.names)]
    result = np.zeros((), dtype=dtype).view(np.recarray)
    for i, (t, y, _dy, a) in enumerate(L):
        result["t"][i, :] = np.squeeze(thin(t, nthin))
        for k in y.dtype.names:
            result[k][i, :] = np.squeeze(thin(y[k], nthin))
        for k in a.dtype.names:
            result[k][i, :] = np.squeeze(thin(a[k], nthin))
    return result.reshape(1)


class Clampable(object):
    """
    :wiki:`Mixin` class for in silico experimental protocols for Bondarenko-like models.
    
    Each protocol returns a list of ``(t, y, dy, a)``, concatenated if ``concatenate=True``.
    
    Models are assumed to have parameters *stim_duration* and *stim_amplitude*, and 
    context managers :meth:`~cvodeint.namedcvodeint.Namedcvodeint.clamp` and 
    :meth:`~cellmlmodels.cellmlmodel.Cellmlmodel.dynclamp`.
    
    The 
    :cellml:`Bondarenko 
    <11df840d0150d34c9716cd4cbdd164c8/bondarenko_szigeti_bett_kim_rasmusson_2004_apical>`,
    :doi:`LNCS <10.1152/ajpheart.00219.2010>` and 
    :cellml:`Ten Tusscher 
    <e946a72663bdf17ef6752980a0232351/tentusscher_noble_noble_panfilov_2004_a>` 
    models have a regular stimulus protocol built in, governed by parameters for 
    period, duration and amplitude. Method :meth:`~cgp.virtexp.elphys.paceable.ap` 
    uses this protocol, computing action potential and calcium transient 
    durations, possibly with rootfinding. The current hack is to set 
    ``stim_duration=inf`` when *stim_amplitude* does not depend on time or its 
    time-dependence is handled inside the protocol function.
    """
    
    def bond_protocols(self, thold=1000, nburnin=10, trest=30000, 
        url="http://ajpheart.physiology.org/content/287/3/H1378.full#F%s"):
        """
        Voltage-clamping protocols from Bondarenko et al. 2004.
        
        Returns an ordered dictionary whose keys are figure numbers
        in :doi:`Bondarenko et al. 2004 <10.1152/ajpheart.00185.2003>`.
        The values are :class:`Bond_protocol` named tuples of 
        ``(varnames, protocol, limits, url)``:
        
        * **varnames**: list of space-delimited strings naming the variables 
          (whether state or algebraic) plotted in each panel
        * **protocol**: input for :meth:`Clampable.pace` or 
          :meth:`Clampable.vecvclamp`. For pacing protocols, 
          stimulus duration and amplitude are taken from the model parameters
        * **limits**: list of original axis limits for time and variable, 
          if applicable. len(limits) == len(varnames)
        * **url**: URL to original figure
        
        :param ms thold: duration at holding potential
        :param nburnin: number of stimuli for pacing protocol
        :param trest: duration of rest before Ca staircase 
            (`figure 19 
            <http://ajpheart.physiology.org/content/287/3/H1378.full#F19>`). 
            Default taken from Bondarenko's ref. 39, Huser et al. 1998, 
            start of Results.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> b = Bond()
        >>> b.bond_protocols(thold=5000)[11]
        Bond_protocol(varnames=['i_Kr'], 
        protocol=[(5000, -80), (1000, array([-70, -60, ...,  60])), (1000, -40)], 
        limits=[[0, 2000, 0, 1]], url='http://...')
        """
        d, a = self.pr.stim_duration, self.pr.stim_amplitude
        return OrderedDict((fignumber, 
            Bond_protocol(varnames, protocol, limits, url % fignumber)) 
            for fignumber, varnames, protocol, limits in [
        (3, ["i_Na"], [(thold, -140), (500, np.arange(-130, 51, 10)), 
            (180, -20)], [[0, 30, -400, 0]]),
        (5, ["i_CaL"], [(thold, -80), (250, np.arange(-70, 41, 10)), 
            (2, -80), (250, 10)], [[0, 500, -8, 0]]), 
        (6, ["Cai Cass", "i_CaL Cai"], [(thold, -80), 
            (200, np.arange(-60, 51, 10))], [[0, 200, 0, 40], None]), 
        (7, ["i_CaL"], [(thold, -80), (250, 0), 
            (np.arange(2, 503, 25), (-90, -80, -70)), 
            (100, 0)], [[0, 800, -8, 0]]), 
        (8, ["i_Kto_f i_Kto_s i_Kr i_Kur i_Kss i_Ks i_K1"], [(thold, -80), 
            (5000, np.arange(-70, 51, 10))], [[0, 5000, 0, 70]]), 
        (9, ["i_Kto_f"], [(thold, -80), (500, np.arange(-100, 51, 10)), 
            (500, 50)], [[0, 1000, 0, 60]]), 
        (10, ["i_Kto_s"], [(thold, -100), (5000, np.arange(-90, 51, 10))], 
            [[0, 2000, 0, 10]]), 
        (11, ["i_Kr"], 
            [(thold, -80), (1000, np.arange(-70, 61, 10)), (1000, -40)], 
            [[0, 2000, 0, 1]]), 
        (12, ["i_Kur", "i_Kss"], 
            [(thold, -100), (5000, np.arange(-90, 51, 10))], 
            [[0, 5000, 0, 15], [0, 5000, 0, 5]]),
        (13, ["i_Kur i_Kss"], 
            [(thold, -100), (5000, np.arange(-60, 61, 10))], [None]),
        (14, ["i_Ks"], [(thold, -80), (5000, np.arange(-70, 51, 10))], 
            [[0, 5000, 0, 0.6]]),
        (15, ["i_K1"], [(thold, -80), (5000, np.arange(-150, -39, 10))], 
            [None]),
        (16, ["V", "i_Kto_f i_Kur i_Kss i_CaL i_Na", 
            "i_NaCa i_NaK i_K1 i_Cab i_Nab"], 
            [(nburnin, 1000, d, a)], [[0, 50, -90, 40], 
            [0, 50, -15, 30], [0, 50, -0.5, 1]]),
        (17, ["V", "Cai"], [(nburnin, (150, 250, 500, 1000, 2000), d, a)], 
            [[0, 50, -90, 40], [0, 140, 0, 0.6]]),
        (18, ["Cai"], [(nburnin, 1000 / np.arange(500, 7001, 500), d, a)], 
            [None]),
        (19, ["Cai"], [(1, trest, d, a), (12, 1300, d, a)], 
            [[0, 15000, 0.1, 0.7]]),
        # Note: J_CaL (uM/ms) is not available, so use i_CaL (pA/pF = V/s)
        # Same for J_NaCa
        (20, ["J_rel i_CaL i_NaCa J_up J_leak"], [(11, 1000, d, a)], 
            [[0, 60, -0.4, 0.4], [0, 1000, -40, 40]]),
        ])
    
    def zhou_protocols(self, thold=1000, nburnin=10, trest=30000, 
        url="http://ajpheart.physiology.org/content/287/3/H1378.full#F%s"):
        """
        Voltage-clamping protocols from Zhou et al. 1998.
        
        .. todo:: Definition lists in ReST have (term, descr) on separate lines
        
        Returns an ordered dictionary whose keys are figure numbers in
        `Zhou et al. 1998 <http://circres.ahajournals.org/cgi/content/full/83/8/806>`_.
        The values are :class:`Bond_protocol` named tuples of 
        (varnames, protocol, limits, url):
        
        * **varnames**: list of space-delimited strings naming the variables 
          (whether state or algebraic) plotted in each panel
        * **protocol**: input for :meth:`Clampable.pace` or :meth:`Clampable.vecvclamp`
          For pacing protocols, stimulus duration and amplitude are taken 
          from the model parameters
        * **limits**: list of original axis limits for time and variable, 
          if applicable. Note that len(limits) == len(varnames)
        * **url**: URL to original figure.
        
        :param ms thold: duration at holding potential
        :param nburnin: number of stimuli for pacing protocol
        :param trest: duration of rest before Ca staircase (figure 19).
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> b = Bond()
        >>> b.zhou_protocols(thold=5000)[2]
        Bond_protocol(varnames=['i_Kto_s'], 
        protocol=[(5000, -50), (200, 40), (5, -50), 
        (100, array([-40, -30, ...,  50])), (100, -20)], limits=[None], 
        url='http://...')
        """
        return OrderedDict((fignumber, 
            Bond_protocol(varnames, protocol, limits, url % fignumber)) 
            for fignumber, varnames, protocol, limits in 
            [(2, ["i_Kto_s"], [(thold, -50), (200, 40), (5, -50), 
            (100, np.arange(-40, 51, 10)), (100, -20)], [None]),
        ])

    def pace(self, protocol, nthin=None):
        """
        Iterator to yield (t, y, dy, a, stats) from specified pacing of a model.
        
        :param protocol: Sequence of (n, period, duration, amplitude)
        :param nthin: number of time-points for each pace (default: no thinning)
        :return: Yields successive named tuples of class Pace with fields 
            ``(t, y, dy, a, stats)``, where *t* is local time, 
            cf. :func:`~cgp.virtexp.elphys.paceable.globaltime` 
            and :func:`~cgp.virtexp.elphys.paceable.localtime`.
        
        Here is an example pacing twice at 70-ms intervals, then three times 
        at 30-ms intervals. Here we store the output of the :meth:`pace`
        iterator in a list so we can reuse it below without recomputing.
        
        .. plot::
            :include-source:
            :context:
            :nofigs:
            
            >>> from cgp.virtexp.elphys.examples import Bond
            >>> b = Bond(reltol=1e-3)
            >>> protocol = [(2, 70, 0.5, -80), (3, 30, 0.5, -80)]
        
        .. plot::
            :context:
            
            for t, y, dy, a, stats in b.pace(protocol):
                plt.plot(t, y.V)
        
        You may concatenate the output with :func:`catrec`. To reuse results
        of lengthy computations, the output of the :meth:`pace` iterator
        can be stored in a list.
        
        .. plot::
            :context:
            :include-source:
            :nofigs:
            
            >>> from cgp.virtexp.elphys.clampable import catrec
            >>> L = list(b.pace(protocol))
            >>> t, y, dy, a, stats = catrec(*L)
            >>> "%5.2f" % y.V.max()
            '33.02'
        
        .. plot::
            :context:
        
            plt.clf()
            plt.plot(t, y.V, t, 100 * y.Cai)
            plt.show()

        The named tuples that :meth:`pace` yields can be unpacked as usual, 
        or you may refer to named fields.
        
        >>> ["%4.2f" % stats["caistats"]["peak"] for t, y, dy, a, stats in L]
        ['0.64', '0.54', '0.53', '0.49', '0.38']
        >>> [(i.t[0], i.t[-1]) for i in L]
        [(0.0, 70.0), (0.0, 70.0), (0.0, 30.0), (0.0, 30.0), (0.0, 30.0)]
        
        The *nthin* argument can reduce the size of output.
        
        >>> all([len(i.y.Cai) == 3 for i in b.pace(protocol, nthin=3)])
        True
        """
        y0 = np.copy(self.y)
        for n, period, duration, amplitude in protocol:
            with self.autorestore(_y=y0, stim_period=period, 
                stim_duration=duration, stim_amplitude=amplitude):
                for _i in range(n):
                    t, y, stats = self.ap()
                    # I could add an option to skip rates_and_algebraic, 
                    # but this takes only 8% of the time for ap anyway.
                    # We could save 30% time by predefining e.g. 100 
                    # time-points instead of letting cvode return ~1500 
                    # points, but the preset temporal resolution might not suit 
                    # the dynamics from another initial state or parameter set.
                    dy, a = self.rates_and_algebraic(t, y)
                    if nthin:
                        t, y, dy, a = [thin(arr, nthin) for arr in t, y, dy, a]
                    yield Pace(t, y, dy, a, stats)
            y0 = y[-1]
    
    def vecpace(self, protocol, nthin=None):
        """
        Vectorized :meth:`~Clampable.pace`.
        
        :param protocol: Sequence of (n, period, duration, amplitude).
            If any n, period, duration or amplitude is a sequence of length > 1, 
            multiple protocols are computed by :func:`ndbcast`.
        :param nthin: Number of time-points for each pulse (default: no thinning).
        :return list: Input and output (protocol_i, [list of Pace]) for each 
            call to :meth:`~Clampable.pace`, one for each unique protocol.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> b = Bond()
        >>> protocol = [(3, (150, 250), 0.25, -80)]
        >>> L = b.vecpace(protocol)
        
        An example using named fields of :class:`Pace` objects.
        
        >>> for proto, paces in L:
        ...     print proto,
        ...     for pace in paces:
        ...         print "%8.3f" % pace.y.V.max(),
        [(3, 150, 0.25, -80)]   31.584 -63.698 -63.112
        [(3, 250, 0.25, -80)]   31.584 -63.252 -62.847
        """
        return [(p, list(self.pace(p, nthin))) for p in ndbcast(*protocol)]

    @contextmanager
    def dynclamp(self, setpoint, R=0.02, V="V", ion="Ki", scale=None):
        """
        Derived model with state value dynamically clamped to a set point.
        
        Input arguments:
        
        * setpoint : target value for state variable
        * R=0.02 : resistance of clamping current
        * V="V" : name of clamped variable
        * ion="Ki" : name of state variable carrying the clamping current
        * scale=None : "ion" per "V", 
          default :math:`Acap * Cm / (Vmyo * F)`, see below
        
        Clamping is implemented as a dynamically applied current that is 
        proportional to the deviation from the set point::
        
            dV/dt = -(i_K1 + ... + I_app)
            I_app = (V - setpoint) / R
        
        Thus, if V > setpoint, then I_app > 0 and serves to decrease dV/dt.
        To account for the charge added to the system, a current proportional 
        to I_app is added to a specified ion concentration, by default Ki. This 
        needs to be scaled according to conductance and other constants.
        The default is as for the Bondarenko model::
         
            scale = Acap * Cm / (Vmyo * F)
            dKi/dt = -(i_K1 + ... + I_app) * scale
                
        Example with voltage clamping of Bondarenko model. Any pre-existing 
        stimulus amplitude is temporarily set to zero. The parameter array is 
        shared between the original and clamped models, and restored on 
        exiting the 'with' block.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> bond = Bond()
        >>> with bond.dynclamp(-140) as clamped:
        ...     t, y, flag = clamped.integrate(t=[0, 10])
        ...     bond.pr.stim_amplitude
        array([ 0.])
        >>> clamped.pr is bond.pr
        True
        >>> bond.pr.stim_amplitude
        array([-80.])
        
        The clamped model has its own instance of the CVODE integrator and 
        state NVector.
        
        >>> (clamped.cvode_mem is bond.cvode_mem, clamped.y is bond.y)
        (False, False)
        
        However, changes in state are copied to the original on exiting the 
        'with' block.
        
        >>> "%7.2f" % bond.yr.V
        '-139.68'
                
        Unlike .clamp(), .dynclamp() does not allow you to change the setpoint 
        inside the "with" block. Instead, just start a new "with" block.
        (Changes to state variables remain on exit from the with block.)
        
        >>> with bond.dynclamp(-30) as clamped:
        ...     t0, y0, flag0 = clamped.integrate(t=[0, 10])
        >>> with bond.dynclamp(-10) as clamped:
        ...     t1, y1, flag1 = clamped.integrate(t=[10, 20])
        >>> np.concatenate([y0[0], y0[-1], y1[0], y1[-1]])["V"].round(2)
        array([-139.68,  -29.96,  -29.96,  -10.34])
        
        Naive clamping with dV/dt = 0 and unspecified clamping current, 
        like .clamp() does, equals the limit as R -> 0 and scale = 0.
        
        >>> with bond.autorestore(V=0):
        ...     with bond.dynclamp(-140, 1e-10, scale=0) as clamped:
        ...         t, y, flag = clamped.integrate(t=[0, 0.1])
        
        Although V starts at 0, it gets clamped to the setpoint very quickly.
        
        >>> y.V[0]
        array([   0.])
        >>> t[y.V.squeeze() < -139][0]
        4.94...e-10
        """
        if scale is None:
            p = self.pr
            scale = p.Acap * p.Cm / (p.Vmyo * p.F)
        
        # Indices to state variables whose rate-of-change will be modified.
        iV = self.dtype.y.names.index(V)
        iion = self.dtype.y.names.index(ion)
        
        def dynclamped(t, y, ydot, f_data):
            """New RHS that prevents some elements from changing."""
            self.f_ode(t, y, ydot, f_data)
            I_app = (y[iV] - setpoint) / R
            ydot[iV] -= I_app
            ydot[iion] -= I_app * scale
            return 0
        
        y = np.array(self.y).view(self.dtype.y)
        
        # Use original options when rerunning the Cvodeint initialization.
        oldkwargs = dict((k, getattr(self, k)) 
            for k in "chunksize maxsteps reltol abstol".split())
        
        pr_old = self.pr.copy()
        
        args, kwargs = self._init_args
        clamped_model = self.__class__(*args, **kwargs)
        Namedcvodeint.__init__(clamped_model, 
            dynclamped, self.t, y, self.pr, **oldkwargs)
        
        # Disable any hard-coded stimulus protocol
        if "stim_amplitude" in clamped_model.dtype.p.names:
            clamped_model.pr.stim_amplitude = 0
        
        try:
            yield clamped_model # enter "with" block
        finally:
            self.pr[:] = pr_old
            for k in clamped_model.dtype.y.names:
                if k in self.dtype.y.names:
                    setattr(self.yr, k, getattr(clamped_model.yr, k))

    def vclamp(self, protocol, nthin=None):
        """
        Iterator to yield t, y, dy, a from a voltage clamp experiment.
        
        :param protocol: Sequence of (duration, voltage) for each pulse
        :param nthin: number of time-points for each pulse (default: no thinning)
        :return: Yields successive named tuples of (t, y, dy, a) 
            where t is local time, cf. 
            :func:`~cgp.virtexp.elphys.paceable.globaltime`.
        
        Here is an example of the P1-P2 protocol used in Figure 3 of 
        Bondarenko et al. 2004.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> b = Bond()
        
        Simulate three intervals: holding potential, P1 pulse, P2 pulse.
        
        >>> L = b.vclamp([(1000, -140), (500, -70), (180, -20)])
        
        Peak (most negative) i_Na in the P1 and P2 intervals:
        
        >>> ["%5.2f" % i.a.i_Na.min() for i in L[1:]]
        ['-0.11', '-99.78']
        
        State and parameters are autorestored after the protocol is finished.
        
        >>> all(b.y == b.model.y0)
        True
        
        Verify that the stimulus current is disabled during clamping.
        
        >>> all(L[0].a.i_stim == 0)
        True
        """
        L = []
        with self.autorestore():
            for duration, voltage in protocol:
                with self.clamp(V=voltage) as clamped:
                    t, y, _flag = clamped.integrate(t=[0, duration])
                    dy, a = self.rates_and_algebraic(t, y)
                if nthin:
                    t, y, dy, a = [thin(arr, nthin) for arr in t, y, dy, a]
                L.append(Trajectory(t, y, dy, a))
        return L
    
    def vargap(self, protocol, nthin=None):
        """
        Variable-gap protocol without duplication of effort.
        
        :param protocol: sequence of (duration, voltage) for 
            (holding, p1, gap, p2) where gap duration and voltage may be 
            vectors.
            If any duration or voltage is a sequence of length > 1, 
            multiple protocols are computed by :func:`pairbcast`.
        :param nthin: thinning output as for vclamp
        :return: Trajectories for p1, gap, p2; the latter in lists.
        
        * The holding trajectory is discarded because it is not of interest.
        * The p1 trajectory is common to all variations of the protocol, 
          so only computed once.
        * Gap trajectories are returned only for the longest gap for each 
          unique gap voltage, because the shorter gap dynamics duplicate 
          the longest up to their endpoint.
        * All p2 trajectories are returned because they depend on previous 
          history.
        
        .. todo:: `vargap` gives blatantly wrong result, will fix later. 
           Workaround: Stick with vecvclamp.
        """
        with self.autorestore():
            # Run holding interval, then P1 pulse, 
            # leaving the P1 trajectory in (t, y).
            for duration, voltage in protocol[:2]:
                with self.clamp(V=voltage) as clamped:
                    t, y, _flag = clamped.integrate(t=[0, duration])
            dy, a = self.rates_and_algebraic(t, y)
            p1 = [Trajectory(t, y, dy, a)]
            # Run gap, spawning P2 pulses at the prescribed times
            p1_duration, _p1_voltage = protocol[1]
            gap_durations, gap_voltages = protocol[2]
            p2_duration, p2_voltage = protocol[3]
            gap = []
            p2 = []
            for gap_voltage in gap_voltages:
                gap_trajectory = []
                with self.autorestore(): # Restore after each gap voltage
                    with self.clamp(V=gap_voltage) as clamped:
                        tspan = np.ones(2) * p1_duration
                        for gap_duration in gap_durations:
                            tspan = np.array([tspan[-1], 
                                              p1_duration + gap_duration])
                            gap_trajectory.append(clamped.integrate(t=tspan))
                            with clamped.autorestore(): # Restore after each p2
                                with clamped.clamp(V=p2_voltage) as p2_clamped:
                                    p2tspan = (tspan[-1] + 
                                               np.array([0, p2_duration]))
                                    t, y, _flag = p2_clamped.integrate(
                                        t=p2tspan)
                            dy, a = self.rates_and_algebraic(t, y)
                            p2.append(Trajectory(t, y, dy, a))
                        # Concatenate and store gap dynamics
                        t, y, _flag = catrec(*gap_trajectory, 
                                             globalize_time=False)
                        dy, a = self.rates_and_algebraic(t, y)
                        gap.append(Trajectory(t, y, dy, a))
        if nthin:
            p1 = [Trajectory(*[thin(arr, nthin) for arr in i]) for i in p1]
            gap = [Trajectory(*[thin(arr, nthin) for arr in i]) for i in gap]
            p2 = [Trajectory(*[thin(arr, nthin) for arr in i]) for i in p2]
        return p1, gap, p2
    
    def vecvclamp(self, protocol, nthin=None, log_exceptions=False):
        """
        Vectorized :meth:`~Clampable.vclamp`.
        
        :param protocol: sequence of (duration, voltage)
            If any duration or voltage is a sequence of length > 1, 
            multiple protocols are computed by pairbcast().
        :param nthin: thinning output as for :meth:`~Clampable.vclamp`
        :param bool log_exceptions: handle any exceptions by logging a warning
        :return: List with input and output (protocol_i, trajectories_i) for 
            each call to :meth:`~Clampable.vclamp`, one for each unique protocol.
        
        >>> from cgp.virtexp.elphys.examples import Bond
        >>> b = Bond(reltol=1e-3)
        >>> protocol = (1000, -140), (500, np.linspace(-80, 40, 4)), (180, -20)
        >>> L = b.vecvclamp(protocol)
        
        The first protocol.
        
        >>> L[0][0]
        [(1000, -140), (500, -80.0), (180, -20)]
        
        An example using named fields of :class:`Trajectory` objects.
        
        >>> for proto, traj in L:
        ...     t1, v1 = proto[1]
        ...     print "%3d: %8.3f" % (v1, traj[1].a.i_Na.min())
        -80:   -0.004
        -40: -175...
          0: -300...
         40:    0.000
        
        State and parameters are autorestored after the protocol is finished.
        
        >>> all(b.y == b.model.y0)
        True
        """
        L = []
        for p in pairbcast(*protocol):
            try:
                L.append((p, self.vclamp(p, nthin)))
            except Exception, _exc:  # pylint: disable=W0703,W0612
                logger.exception("Error in vclamp(%s)", p)
        return L
    
    def bondfig3(self, thold=1000, vhold=-140, 
        t1=500, v1=np.arange(-140, 51, 10), 
        t2=180, v2=-20, plot=True):
        """
        P1-P2 protocol from Fig. 3 of Bondarenko et al.
        """
        L = self.vecvclamp([(thold, vhold), (t1, v1), (t2, v2)])
        peak1, peak2 = [np.array([traj[i].a.i_Na.min() for _proto, traj in L])
            for i in 1, 2]
        peak1 = - peak1 / peak1.min()
        peak2 = peak2 / peak2.min()
        
        if plot:
            from pylab import figure, subplot, plot, axis
            figure()
            subplot(221)
            _h = [plot(tr[1].t, tr[1].a.i_Na, label=pr[1][1]) for pr, tr in L]
            axis(xmax=30)
            subplot(222)
            plot(v1, peak1, '.-')
            subplot(223)
            plot(v1, peak2, '.-')
        
        return L
    
    def bondfig5(self, thold=1000, vhold=-80, t1=250, 
                 v1=np.arange(-70, 41, 10), 
                 t2=2, v2=-80, t3=250, v3=10, plot=True):
        """Fig. 5 of Bondarenko et al."""
        L = self.vecvclamp([(thold, vhold), (t1, v1), (t2, v2), (t3, v3)])
        peak1, peak3 = [np.array([traj[i].a.i_CaL.min() for _proto, traj in L])
            for i in 1, 3]
        peak1 = - peak1 / peak1.min()
        peak3 = peak3 / peak3.min()
        
        if plot:
            from pylab import figure, subplot, plot, axis
            figure()
            subplot(221)
            for v1i, (_proto, traj) in zip(v1, L):
                t, _y, _dy, a = catrec(*traj[1:])
                plot(t - t[0], a.i_CaL, label=v1i)
            axis(ymin=-8, xmax=500)
            subplot(222)
            plot(v1, peak1, '.-')
            axis([-60, 60, -1, 0])
            subplot(223)
            plot(v1, peak3, '.-')
            axis([-80, 40, 0, 1])
        
        return L

    def bondfig6(self, thold=1000, vhold=-80, t1=200, 
                 v1=np.arange(-60, 51, 10), plot=True):
        """Fig. 6 of Bondarenko et al."""
        L = self.vecvclamp([(thold, vhold), (t1, v1)])
        peakCai = np.array([traj[1].y.Cai.max() for _proto, traj in L])
        peaki_CaL = np.array([traj[1].a.i_CaL.max() for _proto, traj in L])
        peakCai = peakCai / peakCai.max()
        peaki_CaL = peaki_CaL / peaki_CaL.max()
        
        if plot:
            from pylab import figure, subplot, plot, axis
            figure()
            subplot(211)
            for _proto, (_, (t, y, _dy, _a)) in L:
                plot(t, y.Cai, t, y.Cass)
            axis(xmax=500)
            subplot(212)
            plot(v1, peakCai, '.-', v1, peaki_CaL, '.-')
        
        return L

    def bondfig7(self, protocol=
        ((1000, -80), (250, 0), (np.arange(2, 503, 25), -80), (100, 0)), 
        plot=True):
        L = self.vecvclamp(protocol)
        if plot:
            from pylab import figure, subplot, plot
            figure()
            subplot(211)
            for _proto, traj in L:
                t, _y, _dy, a = catrec(*traj[1:])
                plot(t - t[0], a.i_CaL)
            # axis(xmax=500)
            # subplot(212)
            # plot(v1, peakCai, '.-', v1, peaki_CaL, '.-')
            # peakCai = np.array([traj[1].y.Cai.max() for proto, traj in L])
        return L

    def nelsonCa(self, protocol=((1000, -75), (4000, 0)), plot=True):
        # "," unpacks 1tuple
        (_proto, (_traj0, traj1)), = self.vecvclamp(protocol)
        
        if plot:
            from pylab import figure, plot, axis
            figure()
            # semilogy(traj1.t, -traj1.a.i_CaL)
            plot(traj1.t, traj1.a.i_CaL)
            axis("tight")
        
        return traj1

def mmfits(L, i=2, k=None, abs_=True):
    """
    Convenience wrapper for applying mmfit() to list returned by vecvclamp().
    
    :param L: list of (proto, traj), where *proto* is a list of 
        (duration, voltage) for each pulse, and *traj* is a list of 
        :class:`Trajectory` named tuples of (t, y, dy, a).
        The first "pulse" is usually at a holding potential.
    :param i: index of "interpulse gap" pulse.
    :param str k: name of field in y or a.
    :param bool abs_: use absolute value of y[k] or a[k]?
    :return: *ymax, xhalf* : Michaelis-Menten parameters for peak y[k] 
        or a[k] in pulse [i+1] vs. duration of interpulse interval [i].
    
    Example: Michaelis-Menten fit of peak i_CaL current vs gap duration::
    
        from cgp.virtexp.elphys.examples import Bond
        b = Bond()
        protocol = (1000, -80), (250, 0), (np.linspace(2, 202, 5), -80), (100, 0)
        with b.autorestore():
            L = b.vecvclamp(protocol)
        mmfits(L, k="i_CaL")
        
        (6.44..., 18.14...)
    
    (This uses a reduced version of the variable-gap protocol of Bondarenko's 
    figure 7. The full version is available as b.bond_protocols()[7].protocol.)
    """
    # Without this assertion, we'd get 
    # AttributeError: 'NotImplementedType' object has no attribute 'max'
    msg = "k must be a single field name of y or a, not %s" % type(k)
    assert isinstance(k, basestring), msg
    tgap = []
    peak = []
    for proto, traj in L:
        duration, _voltage = proto[i]
        _t, y, _dy, a = traj[i+1]
        curr = y[k] if k in y.dtype.names else a[k]
        if abs_:
            curr = np.abs(curr)
        tgap.append(duration)
        peak.append(curr.max())
    return mmfit(tgap, peak)

def mmfit(x, y, rse=False):
    """
    Michaelis-Menten fit, y = ymax x / (x + xhalf).
    
    Passing rse=False also returns relative standard errors for each estimate.
    
    :return ymax: asymptotic value of y as x goes to infinity
    :return xhalf: half-saturation value of x
    :return rse_ymax: relative standard error for *ymax* (if ``rse=True``)
    :return rse_xhalf: relative standard errors for *xhalf* (if ``rse=True``)
    
    For use with results from :meth:`Clampable.vecvclamp`, 
    see :func:`mmfits`.
    
    >>> np.random.seed(0)
    >>> x = np.arange(10.0)
    >>> y =  x / (x + 5) + 0.05 * np.random.random(size=x.shape)
    >>> mmfit(x, y)
    (0.99860..., 4.35779...)
    >>> mmfit(x, y, rse=True)
    (0.99860..., 4.35779..., 0.04162..., 0.09433...)
    
    Note that the estimators are somewhat biased.
    
    >>> np.mean([mmfit(x, x / (x + 5) + 0.05 * np.random.randn(*x.shape)) 
    ...     for i in range(100)], axis=0)
    array([ 1.059...,  5.690...])
    
    (Increasing to 10000 samples gives [ 1.035,  5.401].)
    
    Verify fix for excessive output on error (unwanted dump of source code).
    
    >>> from cgp.utils import rnumpy
    >>> rnumpy.RRuntimeError.max_lines = 0
    >>> mmfit(range(5), range(5))
    (nan, nan)
    
    If the estimate of ymax or xhalf is negative, nans are returned.
    A debug message is also logged.
    
    >>> mmfit(x, -y)
    (nan, nan)
    """
    with roptions(show_error_messages=False, deparse_max_lines=0):
        kwargs = dict(formula="y~ymax*x/(x+xhalf)", data=dict(x=x, y=y), 
            start=dict(ymax=max(y), xhalf=np.mean(x)))
        try:
            fit = r.nls(**kwargs)
        except RRuntimeError, exc:
            s = str(exc.exc).split("\n")[1].strip()
            errmsg = "Michaelis-Menten fit failed with message '%s'. " % s
            errmsg += "Arguments to r.nls() were: %s"
            logger.debug(errmsg, kwargs)
            ymax = xhalf = rse_ymax = rse_xhalf = np.nan
        else:
            coef = r2rec(r.as_data_frame(r.coef(r.summary(fit))))
            ymax, xhalf = coef.Estimate
            rse_ymax, rse_xhalf = coef["Std. Error"] / coef.Estimate
            if (ymax < 0) or (xhalf < 0):
                errmsg = ("Michalis-Menten fit gave negative estimate(s): "
                    "ymax=%s, xhalf=%s. Arguments to r.nls() were: %s")
                logger.debug(errmsg, ymax, xhalf, kwargs)
                ymax = xhalf = rse_ymax = rse_xhalf = np.nan
        return (ymax, xhalf, rse_ymax, rse_xhalf) if rse else (ymax, xhalf)

def decayfits(L, i, k, abs_=True):
    """
    Convenience wrapper for applying decayfit() to list returned by vecvclamp().
    
    :param L: list of (proto, traj), where *proto* is a list of 
        (duration, voltage) for each pulse, and *traj* is a list of 
        :class:`Trajectory` named tuples of (t, y, dy, a).
        The first "pulse" is usually at a holding potential.
    :param i: index of the pulse to fit decay for.
    :param str k: name of field in y or a to fit decay of.
    :param bool abs_: use absolute value of y[k] or a[k]?
    :return vi: voltage...
    :return tau: ...and corresponding tau for pulse [i] for each 
        protocol and trajectory in L.
    
    Example:
    
    >>> from cgp.virtexp.elphys.examples import Bond
    >>> b = Bond()
    >>> L = b.vecvclamp(protocol=[(1000, -140), (500, (-80, -50, 10))])
    >>> decayfits(L, 1, "i_Na")
    ([-80, -50, 10], [nan, 58.5..., 0.48...])
    
    Idiot proofing to avoid cryptic error.
    
    >>> decayfits(L, 1, ["i_Na"])
    Traceback (most recent call last):
    AssertionError: k must be a single field name of y or a
    """
    assert isinstance(k, basestring), "k must be a single field name of y or a"
    vi = []
    tau = []
    for proto, traj in L:
        _duration, voltage = proto[i]
        t, y, _dy, a = traj[i]
        curr = y[k] if k in y.dtype.names else a[k]
        if abs_:
            curr = np.abs(curr)
        vi.append(voltage)
        tau.append(decayfit(t, curr))
    return vi, tau

def decayfit(t, y, p=(0.05, 0.9), prepend_zero=False, rse=False, lm=False):
    """
    Fit exponential decay, y(t) = w exp(-t/tau), to latter part of trajectory.
    
    :param t: time
    :param y: variable, e.g. an ion current
    :param p: proportion of return to initial value, 
        as for action potential duration
    :param bool prepend_zero: add a 0 at the start of y (hack for use with 
        ion currents)
    :param bool rse: return relative standard error of slope (not of *tau*!).
    :param bool lm: Return R object for fitted linear model.
    :return tau: -1/slope for estimated slope of log(y) vs t
    
    tau has dimension 'time' and is analogous to half-life: '1/e'-life = 0.37-life.
    
    For use with results from :meth:`Clampable.vecvclamp`, see :func:`decayfits`.
    
    Trivial example.
    
    >>> t = np.arange(10)
    >>> y = np.exp(-t)
    >>> y[0] = 0
    >>> decayfit(t, y)
    1.0
    
    Relative standard error when adding noise.
    
    >>> np.random.seed(0)
    >>> noisy = y + 0.05 * np.random.random(size=y.shape)
    >>> decayfit(t, noisy, rse=True, p=[0.5, 0.99])
    (1.292..., 0.045...)
    """
    assert len(p) == 2
    if prepend_zero:
        t = np.r_[0, t]
        y = np.r_[0, y]
    stats = apd(t, y, p)
    _ip, i0, i1 = stats["i"]
    # Avoid "Warning message: In is.na(rows) : 
    #        is.na() applied to non-(list or vector) of type 'NULL'"
    if i1 <= i0:
        tau = rse_slope = np.nan
        rlm = None
    else:
        if have_rnumpy:
            with roptions(show_error_messages=False, deparse_max_lines=0):
                try:
                    rlm = r.lm("log(y)~t", data=dict(t=t[i0:i1], 
                                                     y=y[i0:i1].squeeze()))
                    coef = r2rec(r.as_data_frame(r.coef(r.summary(rlm))))
                    _intercept, slope = coef.Estimate
                    _rse_intercept, rse_slope = (coef["Std. Error"] / 
                                                 abs(coef.Estimate))
                    tau = - 1.0 / slope
                except RRuntimeError:
                    tau = rse_slope = np.nan
                    rlm = None
        else:
            from scipy.stats.stats import linregress
            try:
                slope, _intercept, _r_value, _p_value, std_err = linregress(
                    t[i0:i1], np.log(y[i0:i1]).squeeze())
                rse_slope = std_err / np.abs(slope)
                tau = - 1.0 / slope
            except ZeroDivisionError:
                tau = rse_slope = np.nan
                rlm = None
    
    result = (tau,)
    if rse:
        result += (rse_slope,)
    if lm:
        result += (rlm,)
    return result if len(result) > 1 else result[0]

def markovplot(t, y, a=None, names=None, model=None, comp=None, col="bgrcmyk", 
    plotpy=False, plotr=True, newfig=True, **legend_kwargs):
    """
    Plot markov state distribution for ion channel.
    
    :param array_like t: time
    :param recarray y: state
    :param recarray a: algebraics
    :param names: sequence of variable names to include
    :param Cellmlmodel model: Cellmlmodel object
    :param str comp: (partial) name of component
    :param col: sequence of fill colours for stacked area chart
    :param bool plotpy: plot using Python (matplotlib)?
    :param bool plotr: plot using R (ggplot2)?
    :param bool newfig: create a new figure? (if using matplotlib)
    :param ``**legend_kwargs``: passed to matplotlib legend()
    
    If a is None, it will be computed using 
    ``model.rates_and_algebraic(t, y)``.
    If names is ``None``, include all "dimensionless" variables whose 
    component name contains *comp*.
    
    .. ggplot::
       
       from cgp.virtexp.elphys.examples import Bond
       from cgp.virtexp.elphys.clampable import markovplot
       
       bond = Bond()
       t, y, stats = bond.ap()
       p = markovplot(t, y, model=bond, comp="fast_sodium")
    """
    t, i = np.unique(t, return_index=True)
    y = y[i]
    if a is None:
        _dy, a = model.rates_and_algebraic(t, y)
    else:
        a = a[i]
    if names is None:
        # Often a closed state is defined algebraically as 1 minus the others.
        # This puts it apart from other closed states in the list of names 
        # that we generate below. To remedy this, we sort it.
        names = sorted([n for i in "y", "a" 
                        for n, c, u in zip(*model.legend[i]) 
                        if comp in c and u == "dimensionless"])
        # Distribution only makes sense for at least two states
        if len(names) < 2:
            return None
        # Now put any open state(s) first.
        o = [i for i in names if i.startswith("O")]
        no = [i for i in names if not i.startswith("O")]
        names = o + no
    x = np.rec.fromarrays(
        [y[k] if k in y.dtype.names else a[k] for k in names], names=names)
    xc = x.view(float).cumsum(axis=1).view(x.dtype).squeeze()
    if plotr:
        from ...utils.rec2dict import rec2dict
        r["df"] = r.cbind({"t": t}, r.as_data_frame(rec2dict(x)))
        # r["df"] = r.data_frame(t=t, **rec2dict(xc))
        # r["df"] = r("df[c('" + "','".join(["t"] + names) + "')]")
        for pkg in "ggplot2", "reshape", "colorspace":
            try:
                r.library(pkg)
            except RRuntimeError:
                r.install_packages(pkg, repos="http://cran.us.r-project.org")
                r.library(pkg)
        # r.plot(r.df)
        r["xm"] = r.melt(r.df, id_var="t")
        cmd = ("qplot(t, value, fill=variable, geom='area', position='stack', "
            "data=xm) + scale_fill_brewer('State', palette='Set3') + "
            "theme_bw()")
        if comp:
            cmd += "+ labs(title='%s')" % comp
        return r(cmd)
    if plotpy:
        from pylab import figure, fill_between, legend, axis, Rectangle
        if newfig:
            figure()
        prev = 0
        col = [col[i % len(col)] for i in range(len(names))]
        # Workaround for fill_between not being compatible with legend():
        # http://matplotlib.sourceforge.net/users/legend_guide.html
        # #plotting-guide-legend
        symbols = []
        labels = []
        for i, k in enumerate(x.dtype.names):
            kwargs = dict(label=k, facecolor=col[i], edgecolor="none")
            fill_between(t, xc[k], prev, **kwargs)
            prev = xc[k]
            symbols.append(Rectangle((0, 0), 1, 1, **kwargs))
            labels.append(k)
        axis("tight")
        # Reverse to match vertical order
        legend(reversed(symbols), reversed(labels), labelspacing=0, 
               handlelength=1, handletextpad=0.5, borderaxespad=0, 
               **legend_kwargs)

def markovplots(t, y, a=None, model=None):
    """
    Markov plots for all components.
    
    >>> from cgp.virtexp.elphys.examples import Bond
    >>> bond = Bond()
    >>> t, y, stats = bond.ap()
    >>> from cgp.utils.thinrange import thin
    >>> i = thin(len(t), 100)
    
    (Below, the ... ellipsis makes doctest ignore messages that R may print 
    about packages getting loaded. However, doctest output cannot start with 
    ellipsis, so we need to print something else first. Sigh.)
    
    >>> print "Text output ignored:"; L = markovplots(t[i], y[i], model=bond)
    Text output ignored:...
    >>> from cgp.utils.rnumpy import r
    >>> r.windows(record=True) # doctest: +SKIP
    >>> print L # doctest: +SKIP    
    """
    comps = np.unique([c for v in model.legend.values() 
                       for _n, c, _u in zip(*v) if c])
    plots = [markovplot(t, y, model=model, comp=comp) for comp in comps]
    return [(c, p) for c, p in zip(comps, plots) if p]

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
