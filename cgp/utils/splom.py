"""Scatterplot matrix helper functions."""

from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import setp, getp
try:
    from cgp.utils.rnumpy import r
except ImportError:
    import warnings
    warnings.warn("rnumpy not installed, some functions will not work.")

from .thinrange import thin
from .trim import trimpair # we use "trim" as argument name below

def rlist2pydict(x):
    """
    Convert an R named list to a Python dict, recursively.
    
    Factors are returned as strings.
    
    Example:
    
    >>> from cgp.utils.rnumpy import *
    >>> d = dict(i=123, x=0.25, s="test", nested=dict(j=456, t="nested list"))
    >>> r["y"] = rcopy(d)
    >>> print rstr(r.y)
    List of 4
     $ i     : int 123
     $ x     : num 0.25
     $ s     : chr "test"
     $ nested:List of 2
      ..$ j: int 456
      ..$ t: chr "nested list"
    >>> rlist2pydict(r.y) == d
    True
    
    >>> rlist2pydict(r("iris[1:2,]"))
    OrderedDict([('Sepal.Length', array([ 5.1,  4.9])), 
    ('Sepal.Width', array([ 3.5,  3. ])), ('Petal.Length', array([ 1.4,  1.4])), 
    ('Petal.Width', array([ 0.2,  0.2])), ('Species', array(['setosa', 'setosa'], 
    dtype='|S6'))])

    """
    if r.is_null(x):
        return None
    if r.is_factor(x):
        x = r.as_character(x)
    if r.isS4(x):
        slots = [(k, r.slot(x, k)) for k in r.slotNames(x)]
        slots = dict((k, v) for k, v in slots if not r.is_null(v))
        return rlist2pydict(r.list(**slots))
    xnames = r.names(x)
    if r.is_null(xnames):
        result = x[0] if len(x) == 1 else list(x)
        try:
            return np.array(result)
        except StandardError:
            return result
    elif r.is_atomic(x):  # Avoid "$ operator is invalid for atomic vectors"
        return OrderedDict(zip(xnames, x))
    else:
        return OrderedDict((k, rlist2pydict(r.dollar(x, k))) for k in xnames)

def r2rec(x):
    """
    Convert R list to Numpy record array.
    
    >>> r2rec(r.iris)[0:1]
    rec.array([(5.1, 3.5, 1.4, 0.2, 'setosa')], 
    dtype=[('Sepal.Length', '<f8'), ('Sepal.Width', '<f8'), 
    ('Petal.Length', '<f8'), ('Petal.Width', '<f8'), ('Species', '|S10')])
    """
    k, v = zip(*rlist2pydict(x).items())
    return np.rec.fromarrays(v, names=k).view(np.recarray)

try:
    iris = r2rec(r("within(iris, Species <- as.numeric(Species))"))
except NameError:
    iris = None

def spij(m, n, i, j, *args, **kwargs):
    """Subplot specified by nrows, ncols, row, col; row, col start from zero."""
    return plt.subplot(m, n, 1 + i * n + j, *args, **kwargs)

def thinp(h, k, n):
    """Thin property 'k' of handle 'h' to 'n' elements."""
    hk = getp(h, k)[1:-1]
    if len(hk) > 0:
        setp(h, k, thin(hk, n))

# pylint: disable=W0401
def splom(a=None, fun=plt.plot, ntick=3, trim=(0, 0), hkw=(), skw=(),
    *args, **kwargs):
    """
    Scatterplot matrix.
    
    :param recarray a: structured array with named fields 
        (default: ``iris`` dataset from R)
    :param func fun: plot function called as fun(a[i], a[j]) for each i, j
    :param int ntick: number of tick marks on axes
    :param float trim: number or proportion of low and high values to trim 
        (for outliers)
    :param dict hkw: keyword arguments passed to hist()
    :param dict skw: keyword arguments passed to subplots_adjust()    
    
    Further positional or keyword arguments are passed on to ``fun``.
    
    >>> splom(iris, marker="o", color="r", skw=dict(wspace=0.1, hspace=0.1), markersize=6)
    array([[Axes(...),...]], dtype=object)
    """
    if a is None:
        a = iris
    assert a.dtype.names
    my_kwargs = dict(marker=".", color="k", linestyle="none", mew=0)
    my_kwargs.update(**kwargs)
    my_hkw = dict(rwidth=1, ec=[0.8, 0.8, 0.8], fc=[0.8, 0.8, 0.8])
    my_hkw.update(hkw)
    my_skw = dict(wspace=0, hspace=0)
    my_skw.update(skw)
    
    n = len(a.dtype.names)
    setp(plt.gcf(), "facecolor", "w")
    # fig = figure(facecolor="w")
    # ax[i,j] = row i, col j
    ax = np.array([[spij(n, n, i, j, frame_on=False) for j in range(n)] 
        for i in range(n)], object)
    for i, ki in enumerate(a.dtype.names):
        for j, kj in enumerate(a.dtype.names):
            # Finite values only, optionally trimming possible outliers
            aki, akj = trimpair(np.c_[a[ki], a[kj]], *trim).T
            plt.axes(ax[i, j])
            if i > j:
                pts = fun(akj, aki, *args, **my_kwargs)
                setp(pts, antialiased=False)
                # ax[i,j].text(0.5, 0.5, str((kj, ki)), 
                #    transform=ax[i,j].transAxes, ha="center")
            elif i == j:
                # ax[i,j].hist(a[ki], cumulative=False, histtype="step")
                plt.hist(aki, **my_hkw)
            if j == 0:
                plt.ylabel(ki, rotation="horizontal")
            if i == (n - 1):
                plt.xlabel(kj, rotation="vertical")
    # compact the plot
    plt.subplots_adjust(**my_skw)
    # don't show any ticks
    setp([i.xaxis for i in ax.flatten()], "ticks_position", "none")
    setp([i.yaxis for i in ax.flatten()], "ticks_position", "none")
    setp(ax[0, 0], "yticks", []) # upper left is histogram, doesn't need yticks
    setp(ax[:, 1:], "yticks", []) # only col 0 should keep its y tick labels
    setp(ax[:-1, :], "xticks", []) # only row -1 should keep its x tick labels
    # pylint: disable=W0141
    lab = filter(None, [i.get_xticklabels() for i in ax[-1, :]])
    setp(lab, "rotation", "vertical")
    if ntick is not None:
        for i in range(n):
            thinp(ax[i, 0], "yticks", ntick)
            thinp(ax[-1, i], "xticks", ntick)
    
    return ax

def insax(ax, i, j, h, w, **kwargs):
    """
    Axes spanning upper left spij(m,n,i,j) to lower right spij(m,n,i+h-1,j+w-1).
    
    ax is a (nrow, ncol) array of Axes.
    
    Example (using a new figure to avoid interference with any existing figure 
    objects).
    
    >>> fig = plt.figure()
    >>> ax = np.array([plt.subplot(4, 4, i) for i in range(1, 17)], 
    ...     object).reshape(4, 4)
    >>> pos = getp(insax(ax, 1, 1, 2, 2), "position")
    >>> print str(pos).replace("'", "")
    Bbox(array([[ 0.32717391,  0.30869565],...[ 0.69782609,  0.69130435]]))
    """
    (left, _), (_, top) = getp(ax[i, j], "position").get_points()
    (_, bottom), (right, _) = getp(ax[i+h-1, j+w-1], "position").get_points()
    # (left, _), (_, top) = getp(spij(m, n, i, j), "position").get_points()
    # (_, bottom), (right, _) = getp(
    #     spij(m, n, i+h, j+w), "position").get_points()
    width = right - left
    height = top - bottom
    ax = plt.gcf().add_axes([left, bottom, width, height], **kwargs)
    return ax

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: Number of colors.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    
    from http://www.scipy.org/Cookbook/Matplotlib/ColormapTransformations
    """
    from scipy import interpolate
    cdict = cmap._segmentdata.copy()  # pylint: disable=W0212
    # N colors
    colors_i = np.linspace(0, 1., N)
    # N+1 indices
    indices = np.linspace(0, 1., N+1)
    for key in ('red', 'green', 'blue'):
        # Find the N colors
        D = np.array(cdict[key])
        I = interpolate.interp1d(D[:, 0], D[:, 1])
        colors = I(colors_i)
        # Place these colors at the correct indices.
        A = np.zeros((N + 1, 3), float)
        A[:, 0] = indices
        A[1:, 1] = colors
        A[:-1, 2] = colors
        # Create a tuple for the dictionary.
        L = []
        for l in A:
            L.append(tuple(l))
        cdict[key] = tuple(L)
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
