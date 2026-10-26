"""Windows-only flock adapter for FlagGems' test JSON report writer.

The comparison kernels are unmodified. Windows lacks POSIX fcntl; these
single-process tests only use its advisory file-lock interface.
"""

import msvcrt
import os

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(file, operation):
    fd = file if isinstance(file, int) else file.fileno()
    position = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        if operation & LOCK_UN:
            mode = msvcrt.LK_UNLCK
        elif operation & LOCK_NB:
            mode = msvcrt.LK_NBLCK
        else:
            mode = msvcrt.LK_LOCK
        msvcrt.locking(fd, mode, 1)
    finally:
        os.lseek(fd, position, os.SEEK_SET)
