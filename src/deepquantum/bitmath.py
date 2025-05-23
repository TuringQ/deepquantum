"""
Bit-twiddling functions of unsigned integers
"""

from typing import List, Union

import torch


def power_of_2(exp: int) -> int:
    return 1 << exp

def is_power_of_2(number: int) -> bool:
    return (number > 0) and (number & (number - 1) == 0)

def log_base2(number: int) -> int:
    assert is_power_of_2(number)
    return number.bit_length() - 1

# See https://arxiv.org/abs/2311.01512 Alg.1

def get_bit(number: Union[int, torch.Tensor], bit_index: int) -> Union[int, torch.Tensor]:
    return (number >> bit_index) & 1

def flip_bit(number: Union[int, torch.Tensor], bit_index: int) -> Union[int, torch.Tensor]:
    return number ^ (1 << bit_index)

def flip_bits(number: Union[int, torch.Tensor], bit_indices: List[int]) -> Union[int, torch.Tensor]:
    for bit_index in bit_indices:
        number = flip_bit(number, bit_index)
    return number

def insert_bit(number: Union[int, torch.Tensor], bit_index: int, bit_value: int) -> Union[int, torch.Tensor]:
    left = (number >> bit_index) << (bit_index + 1)
    middle = bit_value << bit_index
    right = number & ((1 << bit_index) - 1)
    return left | middle | right

# def create_mask(indices_tensor, target_qubits, target_values):
#     """ Creates a boolean mask where indices match target_values at target_qubits """
#     mask = torch.ones_like(indices_tensor, dtype=torch.bool)
#     for q, val in zip(target_qubits, target_values):
#         mask &= (get_bit(indices_tensor, q) == val)
#     return mask
