#!/usr/bin/env python
"""
qsub a bash wrapper around another script.

Usage: qsubwrap [options] JOBSCRIPT

Example:

== JOBSCRIPT ==
#!/usr/bin/env python
#PBS -l walltime=00:01:00
#PBS -j oe
print "Hello"

generates something like this:

== JOBSCRIPT.sh ==
#!/bin/bash
#PBS -l walltime=00:01:00
#PBS -j oe
./JOBSCRIPT

submits it with qsub, and deletes it when qsub has returned.
"""
import os
import sys
import argparse

from cgp.utils.commands import getstatusoutput # calling qsub

def main():
    """Parse commandline, wrap script, submit with qsub, delete when done."""
    wrapper = "#!/bin/bash\n%s\n%s\n"
    if len(sys.argv) > 1:
        jobscript = sys.argv[-1]
    else:
        print __doc__
        raise RuntimeError("No jobscript specified")
    wrapscript = jobscript + ".sh"
    
    # read PBS directives
    with open(jobscript) as f:
        directives = "\n".join(li.strip() 
            for li in f if li.strip().startswith("#PBS"))
    
    # fill in the wrapper with any directives and the jobscript
    assert not os.path.exists(wrapscript)
    with open(wrapscript, "w") as f:
        f.write(wrapper % (directives, os.path.realpath(jobscript)))
    
    # make the jobscript executable
    status, output = getstatusoutput("chmod u+x %s" % jobscript)
    assert status == 0, output
    
    # submit the wrapper job
    cmd = "qsub %s %s" % (" ".join(sys.argv[1:-1]), wrapscript)
    status, output = getstatusoutput(cmd)
    msg = "qsub returned %s\nCommand: %s\n\nOutput: %s"
    assert status == 0, msg % (status, cmd, output)
    
    os.remove(wrapscript)
    print output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_known_args()
    main()
