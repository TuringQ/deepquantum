"""
Utilities
"""

import time
from collections.abc import Callable
from functools import wraps


def record_time(func: Callable) -> Callable:
    """A decorator that records the running time of a function."""
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        t1 = time.time()
        rst = func(*args, **kwargs)
        t2 = time.time()
        print(f'running time of "{func.__name__}": {t2 - t1}')
        return rst
    return wrapped_function


class Time:
    """A decorator that records the running time of a function."""
    def __init__(self) -> None:
        pass

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            t1 = time.time()
            rst = func(*args, **kwargs)
            t2 = time.time()
            print(f'running time of "{func.__name__}": {t2 - t1}')
            return rst
        return wrapped_function
