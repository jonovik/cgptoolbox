"""Action potential statistics."""
import numpy as np

def apd(time, voltage, p_repol = (0.25, 0.50, 0.75, 0.90),
        interpolate=True, decay_p=None):
    """
    Action potential duration: i, ti, Vi = apd(time, voltage, ...)
    
    Input:
    
    * time: array of time-points
    * voltage: array of voltages
    * p_repol: array of "proportion repolarization" for calculating a.p.d.
    * interpolate: boolean, whether to linearly interpolate between the provided 
      points in time
    * decay_p: argument p_repol to `apd_decayrate()`.

    Output: dict with fields:
    
    * i : indices to original arrays (will not match ti, Vi exactly if interp)
    * ttp : time to peak
    * p_repol : copied from input
    * t_repol : time to p_repol repolarization
    * peak : peak voltage
    
    This simple algorithm assumes that the cell is paced at time==0, 
    and defines "action potential duration" as the latest time after peak that 
    still was not p * 100% repolarized, for each p in p_repol.
    
    A more elaborate algorithm for real experimental data, identifying 
    activation times and screening for noise, is given in:
    Omichi C, Zhou S, Lee M-H, Naik A, Chang C-M, Garfinkel A, Weiss JN, 
    Lin S-F, Karagueuzian HS, Chen P-S (2002) Effects of amiodarone on 
    wave front dynamics during ventricular fibrillation in isolated 
    swine right ventricle. Am J Physiol Heart Circ Physiol 282:1063-1070
    http://dx.doi.org/10.1152/ajpheart.00633.2001
    http://cilit.umb.no/cilit/show.php?record=1414

    Example:
    
    >>> from pprint import pprint
    >>> p_repol = np.r_[0.25, 0.5, 0.75, 0.9]
    >>> time = np.linspace(-np.pi,np.pi,101)
    >>> voltage = np.cos(time)
    
    Without interpolation.
    
    >>> pprint(apd(time, voltage, p_repol, interpolate=False))
    {'amp': 2.0,
     'base': -1.0,
     'decayrate': array([ 1.394...]),
     'i': array([50, 66, 75, 83, 89]),
     'p_repol': array([ 0.25,  0.5 ,  0.75,  0.9 ]),
     'peak': 1.0,
     't_repol': array([ 1.00530965,  1.57079633,  2.07345115,  2.45044227]),
     'ttp': 4.4408920985006262e-16}
    
    Linear interpolation is the default.
    
    >>> pprint(apd(time, voltage, p_repol))
    {'amp': 2.0,
     'base': -1.0,
     'decayrate': array([ 1.388...]),
     'i': array([50, 66, 75, 83, 89]),
     'p_repol': array([ 0.25,  0.5 ,  0.75,  0.9 ]),
     'peak': 1.0,
     't_repol': array([ 1.04693965,  1.57079633,  2.09465301,  2.49855986]),
     'ttp': 4.4408920985006262e-16}
    
    Compare to exact solution for t_repol.
    
    >>> np.testing.assert_allclose(
    ...     np.arccos(1 - 2 * p_repol),
    ...     apd(time, voltage, p_repol)["t_repol"], 
    ...     rtol=1e-3)
    
    Interpolation of t_repol may go haywire on aberrant trajectories.
    If any t_repol does not satisfy
    time[0] <= t_repol[i-1] <= t_repol[i] <= time[-1]
    then that and the following elements are set to NaN.
    
    >>> t = range(9)
    >>> Cai = [6, 7, 8, 15, 16, 16, 14, 9, 8]
    >>> print str(apd(t, Cai)["t_repol"]).lower()
    [ 6.1  6.6  7.5  nan]
    
    Bugfix: This used to return array([45905.25, NaN, NaN, NaN])
    instead of all NaNs.
    
    >>> all(np.isnan(apd(t, 
    ...     [42, 43, 43, 44, 50, 110, 1456, 33790, 61249, 52139])["t_repol"]))
    True
    
    Bugfix: This used to return array([Inf, NaN, NaN, NaN])
    instead of all NaNs.
    
    >>> all(np.isnan(apd(t, 
    ...     [43, 43, 44, 50, 110, 1456, 33790, 61249, 52139])["t_repol"]))
    True
    """
    # standardize input arguments
    time = np.asanyarray(time).squeeze()
    voltage = np.asanyarray(voltage).squeeze()
    p_repol = np.atleast_1d(p_repol)
    # landmarks in the action potential
    base_v = voltage[0]
    peak_i = voltage.argmax()
    peak_t = time[peak_i]
    peak_v = voltage[peak_i]
    threshold = peak_v - p_repol * (peak_v - base_v)
    # Use poor man's interpolation, because 
    # "import scipy.interpolate" currently fails under compython/2.5 on Titan.
    i0, i1 = np.ogrid[0:len(p_repol), 0:len(time)]
    repolarized = (i1>peak_i) & (voltage[i1]<threshold[i0])
    repol_diff = np.diff(repolarized, axis=1)
    # Repolarization occurred between time-point repol_i and repol_i + 1
    repol_i = repol_diff.argmax(axis=1)
    if interpolate:
        # linear interpolation
        with np.errstate(all="ignore"):
            t_repol = (time[repol_i] +
                (threshold - voltage[repol_i]) *
                (time[repol_i + 1] - time[repol_i]) /
                (voltage[repol_i + 1] - voltage[repol_i]))
    else:
        t_repol = time[repol_i]
    
    # sanity checking
    prev = t_repol[0]
    for i, t in enumerate(t_repol):
        if not (time[0] <= prev <= t <= time[-1]):
            t_repol[i:] = np.nan
            break
        prev = t
    
    result = dict(i=np.r_[peak_i, repol_i], base=base_v, ttp=peak_t, 
        peak=peak_v, amp=peak_v - base_v, p_repol=p_repol, t_repol=t_repol)
    result["decayrate"] = apd_decayrate(result, decay_p)
    return result

def apd_decayrate(stats, p=None):
    """
    Compute decay rate of recovery phase, from statistics of action potential.
    
    The decay rate is computed as - (diff log (v - base)) / (diff t) 
    = - (diff log (1-p)) / (diff t).
    The difference is taken between p[0] * 100% and p[1] * 100% recovery.
    If p is None, it defaults to the first and last elements of 
    stats["p_repol"].
    
    >>> t = np.linspace(0, 1, 101)
    >>> v = 1 + 2 * np.exp(-3 * t)
    >>> v[0] = 1
    >>> stats = apd(t, v)
    >>> apd_decayrate(stats).round(3)
    array([ 3.])
    
    >>> time = np.linspace(-np.pi,np.pi,101)
    >>> voltage = np.cos(time)
    >>> stats = apd(time, voltage)
    >>> apd_decayrate(stats)
    array([ 1.388...])
    """
    if p is None:
        p2 = stats["p_repol"][0]
        p3 = stats["p_repol"][-1]
    else:
        p2, p3 = p
    t2, t3 = [stats["t_repol"][stats["p_repol"] == pi] for pi in p2, p3]
    with np.errstate(all="ignore"):  # silence NaN-related errors
        difflnv = np.log(1 - p3) - np.log(1 - p2)
        result = - difflnv / (t3 - t2)
        return result if (result > 0) else np.nan
