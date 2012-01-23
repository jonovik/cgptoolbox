"""Tests for :mod:`cgp.utils.poormanslock`."""
# pylint: disable=W0603, C0111

import unittest
import logging
import os
import signal

from ..utils.poormanslock import Lock, log

import tempfile
import shutil

dtemp = None
olddir = os.getcwd()

def setup():
    global dtemp
    dtemp = tempfile.mkdtemp()
    os.chdir(dtemp)

def teardown():
    os.chdir(olddir)
    shutil.rmtree(dtemp)

def _test_reuse():
    """
    Allow reuse of a lock, rather than having to construct it anew each time.
    
    (Known failure.)
    """
    lock = Lock()
    for _i in range(2):
        with lock:
            assert os.path.exists(lock.lockname)
        assert not os.path.exists(lock.lockname)

msg = "signal.alarm() not available, timeout won't work"

@unittest.skipIf(not hasattr(signal, "alarm"), msg)
def test_timeout():
    """Test raising of "IOError: Timed out waiting to acquire lock"."""
    oldlevel = log.level
    try:
        # suppress error message for next test
        log.setLevel(logging.CRITICAL)
        # Existing lockfile causes Lock to wait until timeout
        _f = open("lock","w")
        try:
            with Lock(max_wait=2):
                pass
        except IOError:
            pass
        else:
            raise Exception("Lock failed to time out")
        os.remove("lock")  # removing the lockfile allows normal operation
        with Lock():
            pass  # no error
    finally:
        log.setLevel(oldlevel)
