"""Functions for modelling dose-response relationships in sigmoid models"""
from __future__ import division
import scipy
import numpy as np

def Hill(x, theta, p):
    """
    The Hill function (Hill, 1910) gives a sigmoid increasing dose-response
    function crossing 0.5 at the threshold theta with a steepness determined by
    the Hill coefficient p.

    :param theta: threshold parameter.
    :param p: Hill coefficient.
    :return: function value in the range [0,1].

    .. plot::
        :width: 400
        :include-source:

        import numpy as np
        from matplotlib import pyplot as plt
        from cgp.sigmoidmodels.doseresponse import Hill

        x = np.arange(0, 10, 0.1)
        y1 = [Hill(X,5,2) for X in x]
        y2 = [Hill(X,5,10) for X in x]
        plt.plot(x,y1,x,y2)
        plt.xlabel('x')
        plt.ylabel('Hill(x,5,p)')
        plt.legend(('p=2','p=10'))

    Example usage::

        >>> Hill(5,5,1)
        0.5
        >>> Hill(10, 5, 5)
        0.96969...

    References:

    * Hill AV (1910)
      `The possible effects of the aggregation of the molecules of haemoglobin
      on its dissociation curves
      <http://jp.physoc.org/content/40/supplement/i.full.pdf+html>`_ J. Physiol.
      40 (Suppl): iv-vii.
    """
    if x <= 0:
        return 0.0
    elif theta == 0:
        return 1
    else:
        return (x > 0) * (x ** p / (x ** p + theta ** p))

def nonmono(x, mu, sigma):
    """
    Bell-shaped gene regulation function

    Normal probability distribution scaled to have maximum 1. Used as the non-
    monotonic gene regulation function in Gjuvsland et al. (2011).

    Example usage:

    >>> nonmono(5, 7, 1)
    0.1353352832366127

    Reference:

    * Gjuvsland AB, Vik JO, Wooliams JA, Omholt SW (2011)
      `Order-preserving principles underlying genotype-phenotype maps ensure
      high additive proportions of genetic variance <http://onlinelibrary.wiley.
      com/doi/10.1111/j.1420-9101.2011.02358.x/full>`_ Journal of Evolutionary
      Biology 24(10):2269-2279.
    """
    return(scipy.exp((-(x - mu) ** 2) / (2 * sigma ** 2)))

def R_logic(Z1, Z2, index, name=False):
    """
    Output the continuous equivalent of 1 of 16 boolean functions of Z1 and Z2.

    :param Z1: scalar in [0, 1]
    :param Z2: scalar in [0, 1]
    :param index: integer between 1 and 16 indicating Boolean function
        (see table below)
    :param names: Boolean whether to return function value (default) or
        function name (names=True)
    :return: function value in the range [0, 1].

    Table of Boolean functions:

    ===== ===================================   ====================================
    index Boolean function                      Algebraic function
    ===== ===================================   ====================================
    1     AND(X1,X2)                            Z1Z2
    2     AND(X1,NOT(X2))                       Z1(1-Z2)
    3     AND(NOT(X1),X2)                       (1-Z1)Z2
    4     AND(NOT(X1),NOT(X2))                  (1-Z1)(1-Z2)
    5     X1                                    Z1
    6     X2                                    Z2
    7     NOT(X1)                               1-Z1
    8     NOT(X2)                               1-Z2
    9     OR(X1,X2)                             Z1+Z2-Z1Z2
    10    OR(NOT(X1),NOT(X2))                   1-Z1Z2
    11    OR(AND(X1,X2),AND(NOT(X1),NOT(X2)))   1-Z1-Z2+Z1Z2+Z1^2Z2+Z1Z2^2-Z1^2Z2^2
    12    OR(AND(X1,NOT(X2)),AND(X1,NOT(X2)))   Z1+Z2-3Z1Z2+Z1^2Z2+Z1Z2^2-Z1^2Z2^2
    13    OR(X1,NOT(X2))                        1-Z2+Z1Z2
    14    OR(NOT(X1),X2))                       1-Z1+Z1Z2
    15    0 (always off)                        0
    16    1 (always on)                         1
    ===== ===================================   ====================================

    Example - Create boolean truth tables for all 16 boolean functions::
      
      >>> X1 = [0, 0, 1, 1]
      >>> X2 = [0, 1, 0, 1]
      >>> [(R_logic([], [], ind, name=True), [R_logic(x1, x2, ind)
      ...   for x1, x2 in zip(X1, X2)]) for ind in range(1, 17)]
      [('AND(X1,X2)', [0, 0, 0, 1]),
      ('AND(X1,NOT(X2))', [0, 0, 1, 0]),
      ('AND(NOT(X1),X2)', [0, 1, 0, 0]),
      ('AND(NOT(X1),NOT(X2))', [1, 0, 0, 0]),
      ('X1', [0, 0, 1, 1]),
      ('X2', [0, 1, 0, 1]),
      ('NOT(X1)', [1, 1, 0, 0]),
      ('NOT(X2)', [1, 0, 1, 0]),
      ('OR(X1,X2)', [0, 1, 1, 1]),
      ('OR(NOT(X1),NOT(X2))', [1, 1, 1, 0]),
      ('OR(AND(X1,X2),AND(NOT(X1),NOT(X2)))', [1, 0, 0, 1]),
      ('OR(AND(X1,NOT(X2)),AND(X1,NOT(X2)))', [0, 1, 1, 0]),
      ('OR(X1,NOT(X2))', [1, 0, 1, 1]),
      ('OR(NOT(X1),X2))', [1, 1, 0, 1]),
      ('0', [0, 0, 0, 0]),
      ('1', [1, 1, 1, 1])]
    """

    if name:
        func = dict({'1': 'AND(X1,X2)',
        '2': 'AND(X1,NOT(X2))',
        '3': 'AND(NOT(X1),X2)',
        '4': 'AND(NOT(X1),NOT(X2))',
        '5': 'X1',
        '6': 'X2',
        '7': 'NOT(X1)',
        '8': 'NOT(X2)',
        '9': 'OR(X1,X2)',
        '10': 'OR(NOT(X1),NOT(X2))',
        '11': 'OR(AND(X1,X2),AND(NOT(X1),NOT(X2)))',
        '12': 'OR(AND(X1,NOT(X2)),AND(X1,NOT(X2)))',
        '13': 'OR(X1,NOT(X2))',
        '14': 'OR(NOT(X1),X2))',
        '15': '0',
        '16': '1'})
        return func[str(np.int(index))]
    else:
        func = dict({'1': 'Z1*Z2',
        '2': 'Z1*(1-Z2)',
        '3': '(1-Z1)*Z2',
        '4': '(1-Z1)*(1-Z2)',
        '5': 'Z1',
        '6': 'Z2',
        '7': '1-Z1',
        '8': '1-Z2',
        '9': 'Z1+Z2-Z1*Z2',
        '10': '1-Z1*Z2',
        '11': '1-Z1-Z2+Z1*Z2+Z1**2*Z2+Z1*Z2**2-Z1**2*Z2**2',
        '12': 'Z1+Z2-3*Z1*Z2+Z1**2*Z2+Z1*Z2**2-Z1**2*Z2**2',
        '13': '1-Z2+Z1*Z2',
        '14': '1-Z1+Z1*Z2',
        '15': '0',
        '16': '1'})
        return eval(func[str(np.int(index))])