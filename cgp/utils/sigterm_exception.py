"""
Raise Python exception on SIGTERM signal.

Batch job scripts that exceed their time limit will be killed with a `SIGTERM
<http://linux.die.net/man/1/kill>`_ signal. By default, this stops execution 
immediately, bypassing any `finally 
<http://docs.python.org/reference/compound_stmts.html#finally>`_ clauses or 
`with statement 
<http://docs.python.org/reference/compound_stmts.html#with with statement>`_
__exit__ handlers. This can fail to flush buffers, or leave lock files or 
otherwise fail to release resources, so that one timed-out job blocks other 
running jobs.

A simple workaround is to change the signal handler for SIGTERM to a Python 
function that raises an exception. If you have coded carefully using 
`with statements
<http://docs.python.org/reference/compound_stmts.html#with>`_ and 
`try..finally 
<http://docs.python.org/reference/compound_stmts.html#finally>`_ to clean up 
and release resources, no other modification is required. There's no need to 
catch the exception; its only purpose is to allow Python's cleanup mechanisms 
to complete.
"""
import signal
 
def term(signum, frame):
    raise SystemExit("Received TERM signal")
 
signal.signal(signal.SIGTERM, term)
