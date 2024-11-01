"""
some math function
"""
from typing import Any, List, Optional, Tuple, Union
import torch

def list_xor(list1, list2):
    return list(set(list1) ^ set(list2))