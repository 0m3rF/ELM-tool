"""
File locking utilities for safe concurrent access to shared configuration files.

This module provides file locking mechanisms to prevent race conditions when
multiple test workers or processes access shared configuration files like
environments.ini and masking.json.
"""

import os
import time
import contextlib
from typing import Optional


class FileLock:
    """
    A simple file-based lock implementation for cross-process synchronization.
    
    This lock uses a lock file to coordinate access to shared resources.
    It's suitable for use with pytest-xdist parallel test execution.
    """
    
    def __init__(self, lock_file: str, timeout: float = 10.0, check_interval: float = 0.1):
        """
        Initialize the file lock.
        
        Args:
            lock_file: Path to the lock file
            timeout: Maximum time to wait for the lock (seconds)
            check_interval: Time between lock acquisition attempts (seconds)
        """
        self.lock_file = lock_file
        self.timeout = timeout
        self.check_interval = check_interval
        self._lock_fd: Optional[int] = None
    
    def acquire(self) -> bool:
        """
        Acquire the lock.
        
        Returns:
            bool: True if lock was acquired, False if timeout occurred
        """
        start_time = time.time()
        
        while True:
            try:
                # Try to create the lock file exclusively
                # O_CREAT | O_EXCL ensures atomic creation
                self._lock_fd = os.open(
                    self.lock_file,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR
                )
                return True
            except FileExistsError:
                # Lock file exists, check timeout
                if time.time() - start_time >= self.timeout:
                    return False
                time.sleep(self.check_interval)
            except Exception:
                # Other errors, fail to acquire
                return False
    
    def release(self):
        """Release the lock."""
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
                self._lock_fd = None
            except Exception:
                pass
        
        try:
            os.remove(self.lock_file)
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock on {self.lock_file} within {self.timeout} seconds")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


@contextlib.contextmanager
def file_lock(file_path: str, timeout: float = 10.0):
    """
    Context manager for file locking.
    
    Usage:
        with file_lock('/path/to/file.ini'):
            # Perform operations on the file
            pass
    
    Args:
        file_path: Path to the file to lock
        timeout: Maximum time to wait for the lock (seconds)
    
    Yields:
        None
    
    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    lock_file = f"{file_path}.lock"
    lock = FileLock(lock_file, timeout=timeout)
    
    try:
        with lock:
            yield
    finally:
        pass


def safe_read_config(file_path: str, read_func, timeout: float = 10.0):
    """
    Safely read a configuration file with locking.
    
    Args:
        file_path: Path to the configuration file
        read_func: Function to call to read the file (should take file_path as argument)
        timeout: Maximum time to wait for the lock (seconds)
    
    Returns:
        The result of read_func
    """
    with file_lock(file_path, timeout=timeout):
        return read_func(file_path)


def safe_write_config(file_path: str, write_func, timeout: float = 10.0):
    """
    Safely write a configuration file with locking.
    
    Args:
        file_path: Path to the configuration file
        write_func: Function to call to write the file (should take file_path as argument)
        timeout: Maximum time to wait for the lock (seconds)
    
    Returns:
        The result of write_func
    """
    with file_lock(file_path, timeout=timeout):
        return write_func(file_path)

