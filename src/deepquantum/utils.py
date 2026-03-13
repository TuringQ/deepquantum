"""Utilities"""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch

import deepquantum as dq


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


def apply_complex_fix(fn: Any, tensors_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the function to the tensors in the dictionary and convert the result to complex dtype."""
    first_tensor = next(iter(tensors_dict.values()))
    probe = fn(torch.empty(0, dtype=first_tensor.real.dtype, device=first_tensor.device))
    target_dtype = dq.dtype_map.get(probe.dtype, probe.dtype)
    return {name: tensor.to(probe.device, target_dtype) for name, tensor in tensors_dict.items()}
