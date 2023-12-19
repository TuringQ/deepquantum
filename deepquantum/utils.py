"""
Utilities
"""

import time
from functools import wraps
from typing import Callable


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


class Time(object):
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
