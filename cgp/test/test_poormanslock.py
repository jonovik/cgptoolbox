"""Tests for :mod:`cgp.utils.poormanslock`."""

from ..utils.poormanslock import os, Lock

def test_reuse():
    """
    Allow reuse of a lock, rather than having to construct it anew each time.
    """
    lock = Lock()
    for _i in range(2):
        with lock:
            assert os.path.exists(lock.lockname)
        assert not os.path.exists(lock.lockname)
