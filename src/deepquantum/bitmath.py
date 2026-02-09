"""
Bit-twiddling functions of unsigned integers
"""

import torch


def power_of_2(exp: int) -> int:
    return 1 << exp


def is_power_of_2(number: int) -> bool:
    return (number > 0) and (number & (number - 1) == 0)


def log_base2(number: int) -> int:
    assert is_power_of_2(number)
    return number.bit_length() - 1


# See https://arxiv.org/abs/2311.01512 Alg.1
def get_bit(number: int | torch.Tensor, bit_index: int) -> int | torch.Tensor:
    return (number >> bit_index) & 1


def flip_bit(number: int | torch.Tensor, bit_index: int) -> int | torch.Tensor:
    return number ^ (1 << bit_index)


def flip_bits(number: int | torch.Tensor, bit_indices: list[int]) -> int | torch.Tensor:
    for bit_index in bit_indices:
        number = flip_bit(number, bit_index)
    return number


def insert_bit(number: int | torch.Tensor, bit_index: int, bit_value: int) -> int | torch.Tensor:
    left = (number >> bit_index) << (bit_index + 1)
    middle = bit_value << bit_index
    right = number & ((1 << bit_index) - 1)
    return left | middle | right


def all_bits_are_one(number: int, bit_indices: list[int]) -> bool:
    for i in bit_indices:
        if not get_bit(number, i):
            return False
    return True


def get_bit_mask(bit_indices: list[int]) -> int:
    return flip_bits(0, bit_indices)
