"""
Poor man's file locking: Create exclusively-locked dummy file, delete when done.

Statements inside a "with Lock():" block are executed while other tasks pause.
The constructor, Lock(lockname="lock", retry_delay=0.1, max_wait=30)
creates a dummy file (called "lock" by default), but only if it does not 
already exist. If the file already exists, the constructor will wait 
"retry_delay" seconds before retrying. Waiting too long ("max_wait") triggers 
an IOError exception.


Typical usage:

from __future__ import with_statement
from poormanslock import Lock

with Lock():
    pass # <insert something to do while other tasks wait>

with Lock(lockname="locked.txt", retry_delay=0.2, max_wait=10):
    pass # <insert something to do here>


To turn on debug-level logging:

>>> import poormanslock, logging
>>> poormanslock.log.setLevel(logging.DEBUG)                    # doctest: +SKIP


Doctests:

>>> from __future__ import with_statement
>>> import signal
>>> if hasattr(signal, "alarm"):
...     log.setLevel(logging.CRITICAL) # suppress error message for next test
...     f = open("lock","w") # existing lockfile causes Lock to wait until timeout
...     with Lock(max_wait=2):
...         pass
... else:
...     print "signal.alarm() not available, timeout won't work"
Traceback (most recent call last):
    ...
IOError: Timed out waiting to acquire lock
>>> if hasattr(signal, "alarm"):
...     import os
...     os.remove("lock") # removing the lockfile allows normal operation
...     with Lock():
...         pass
>>> # no error

@Todo: Allow reuse of a lock, rather than having to construct it anew each time.

>>> lock = Lock()
>>> os.path.exists(lock.lockname)
False
>>> with lock:
...     os.path.exists(lock.lockname)
True
>>> os.path.exists(lock.lockname)
False
>>> with lock:
...     os.path.exists(lock.lockname)
True
>>> os.path.exists(lock.lockname)
False
"""
from __future__ import with_statement
import os # os.remove - delete lockfile
import time # time.sleep - wait between attempts to create lockfile
import signal # signal.alarm - raise exception if things take too long
import logging # logging facilities, useful for debugging
from random import random

log = logging.getLogger("poormanslock")
log.addHandler(logging.StreamHandler())
# tab-delimited format string, see http://docs.python.org/library/logging.html#formatter-objects
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
    
    def __init__(self, lockname="lock", retry_delay=0.1, max_wait=30):
        """Create file "lockname" if not exists, retry until timeout if needed"""
        self.lockname = lockname # name of lockfile (needed by os.remove())
        self.retry_delay = retry_delay
        self.max_wait = max_wait
        if hasalarm:
            signal.signal(signal.SIGALRM, _timeout) # set up handler

    def __enter__(self):
        """Enter context of with statement"""
        if hasalarm:
            signal.alarm(self.max_wait)
        self.fd = None # file descriptor to lockfile (needed by os.close())
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
            signal.alarm(0) # defuse the timer
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context of with statement, closing and removing lockfile"""
        os.close(self.fd)
        os.remove(self.lockname)
        log.debug("Released lock")
        return False

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

"""
Copyright 2008 Jon Olav Vik <jonovik@gmail.com>

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see <http://www.gnu.org/licenses/>.
"""
