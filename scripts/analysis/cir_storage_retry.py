"""Scoped, bounded EIO retries for unchanged CIR checkpoint byte operations."""
from __future__ import annotations
import errno
import functools
import time


def retry_eio(operation, *, delays=(1, 2, 4, 8), sleep=time.sleep):
    @functools.wraps(operation)
    def wrapped(*args, **kwargs):
        for attempt in range(len(delays) + 1):
            try:
                return operation(*args, **kwargs)
            except OSError as error:
                if error.errno != errno.EIO or attempt == len(delays):
                    raise
                print(f'CIR_STORAGE_EIO_RETRY operation={operation.__name__} attempt={attempt + 1}', flush=True)
                sleep(delays[attempt])
    return wrapped


def install():
    from new_interest.training import checkpoint_persistence as storage
    if getattr(storage, '_cir_eio_retry_installed', False):
        return
    # These operations are content-only, atomic or read-only. Never retry training,
    # optimizer updates, hash validation failures, or non-EIO exceptions.
    storage._sha256_file = retry_eio(storage._sha256_file)
    storage._copy_file_bytes = retry_eio(storage._copy_file_bytes)
    storage._cir_eio_retry_installed = True
