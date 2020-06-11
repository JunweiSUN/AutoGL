import os
import sys
import time
import threading
import signal

time_budget, pid = int(sys.argv[1]), int(sys.argv[2])

def raise_timeout_exception(pid_to_kill):
    """
    Helper function to inform the main process
    that time has ran out.
    Parameters:
    ----------
    pid_to_kill: int
        the pid of main process
    ----------
    """
    os.kill(pid_to_kill, signal.SIGTSTP)

# start a timer for timing.
timer = threading.Timer(time_budget, raise_timeout_exception, [pid])
timer.start()
