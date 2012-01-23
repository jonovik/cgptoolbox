"""
Poor man's file locking: Create exclusively-locked dummy file, delete when done.

Statements inside a "with Lock():" block are executed while other tasks pause.
The constructor, Lock(lockname="lock", retry_delay=0.1, max_wait=30)
creates a dummy file (called "lock" by default), but only if it does not 
already exist. If the file already exists, the constructor will wait 
"retry_delay" seconds before retrying. Waiting too long ("max_wait") triggers 
an IOError exception.


Typical usage:

from poormanslock import Lock

with Lock():
    pass # <insert something to do while other tasks wait>

with Lock(lockname="locked.txt", retry_delay=0.2, max_wait=10):
    pass # <insert something to do here>


To turn on debug-level logging:

>>> import poormanslock, logging
>>> poormanslock.log.setLevel(logging.DEBUG)                    # doctest: +SKIP
"""
import os # os.remove - delete lockfile
import time # time.sleep - wait between attempts to create lockfile
import signal # signal.alarm - raise exception if things take too long
import logging # logging facilities, useful for debugging
from random import random

log = logging.getLogger("poormanslock")
log.addHandler(logging.StreamHandler())
# tab-delimited format string, see 
# http://docs.python.org/library/logging.html#formatter-objects
fmtstr = "%(" + ")s\t%(".join(
    "asctime levelname name lineno process message".split()) + ")s"
log.handlers[0].setFormatter(logging.Formatter(fmtstr))

# log.warning("Test warning") # output test warning on import 

# Check if we have signal.alarm (only available on Unix)
hasalarm = hasattr(signal, "alarm")
if not hasalarm:
    import warnings
    warnings.warn("signal.alarm() not available, timeout won't work")    

def _timeout(signum, frame):
    """Signal handler for Lock timeout"""
    message = "Timed out waiting to acquire lock"
    log.error(message)
    raise IOError(message)

class Lock(object):
    """
    Poor man's file locking: Create exclusively-locked dummy file, delete when done.
    
    Statements inside a "with Lock():" block are executed while other tasks pause.
    The constructor, Lock(lockname="lock", retry_delay=0.1, max_wait=30)
    creates a dummy file (called "lock" by default), but only if it does not 
    already exist. If the file already exists, the constructor will wait 
    "retry_delay" seconds before retrying. Waiting too long ("max_wait") triggers 
    an IOError exception.
    """
        
    def __init__(self, lockname="lock", retry_delay=0.1, max_wait=30):
        """
        Create file "lockname" if not exists, retry until timeout if needed.
        """
        self.lockname = lockname # name of lockfile (needed by os.remove())
        self.retry_delay = retry_delay
        self.max_wait = max_wait
        self.fd = None # file descriptor to lockfile (needed by os.close())
        if hasalarm:
            # Set up handler
            signal.signal(signal.SIGALRM, _timeout)  # @UndefinedVariable

    def __enter__(self):
        """Enter context of with statement"""
        if hasalarm:
            signal.alarm(self.max_wait)  # @UndefinedVariable
        while self.fd is None:
            try:
                # open file for exclusive access, raise exception if it exists
                log.debug("Requesting lock")
                self.fd = os.open(self.lockname, os.O_EXCL | os.O_CREAT)
                log.debug("Acquired lock")
            except OSError:
                log.debug("Failed to acquire lock")
                # wait before trying again
                time.sleep((0.5 + 0.5 * random()) * self.retry_delay)
        if hasalarm:
            # Defuse the timer
            signal.alarm(0)  # @UndefinedVariable
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context of with statement, closing and removing lockfile"""
        os.close(self.fd)
        os.remove(self.lockname)
        log.debug("Released lock")
        return False

if __name__ == "__main__":
    import doctest
    doctest.testmod()
