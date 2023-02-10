from functools import wraps
import time


def record_time(func):
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        t1 = time.time()
        rst = func(*args, **kwargs)
        t2 = time.time()
        print(f'running time of "{func.__name__}": {t2 - t1}')
        return rst
    return wrapped_function


class Time(object):
    def __init__(self) -> None:
        pass

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            t1 = time.time()
            rst = func(*args, **kwargs)
            t2 = time.time()
            print(f'running time of "{func.__name__}": {t2 - t1}')
            return rst
        return wrapped_function