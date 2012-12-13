#!/usr/bin/env python
# Time limit HH:mm:ss
#PBS -lwalltime=00:05:00
# Join standard error into output
#PBS -j oe
"""Usage demo of utils.arrayjob. Submit with: python -m utils.test_arrayjob."""

# pylint: disable=W0614,W0612

# This import will chdir to PBS_O_WORKDIR, where job was submitted 
from cgp.utils.arrayjob import *  # @UnusedWildImport
set_NID(16) # ID range will be 0, 1, ..., get_NID()-1 

# No boilerplate code below this line! 

import numpy as np  #@Reimport

# Global variables
infile = "input.npy"
outfile = "output.npy"
n = 43 # Number of workpieces

@qopt("-l walltime=00:01:00") # can add stage-specific options
def setup():
    """Initialize data."""
    np.save(infile, np.arange(n))
    # for real work, the fastest way to preallocate an output file is 
    # np.lib.format.open_memmap()

def work():
    """Process chunk ID of workpieces."""
    x = memmap_chunk(infile) # numpy array, memory-mapped to chunk of file
    x += 1 # automatically written when garbage collected

def wrapup():
    """Summarize results."""
    np.save(outfile, sum(np.load(infile)))

if __name__ == "__main__":
    # Specify simulation stage sequence, and whether each stage is parallel
    arun(setup, par(work), wrapup, loglevel="DEBUG")


# Calling arun() only if running as the main script (__name__ == "__main__")
# allows you to make preliminary reports while your results are being computed.
#
# >>> from utils.test_arrayjob import report; report()
def report():
    """Report results, perhaps while they are being computed."""
    print np.load(outfile) # read-only by default
