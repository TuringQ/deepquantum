"""Bit-twiddling functions of unsigned integers"""

import torch


def power_of_2(exp: int) -> int:
    """Calculate 2 raised to the power of the given exponent."""
    return 1 << exp


def is_power_of_2(number: int) -> bool:
    """Check if a number is a power of 2."""
    return (number > 0) and (number & (number - 1) == 0)


def log_base2(number: int) -> int:
    """Calculate the base-2 logarithm of a power-of-2 number."""
    assert is_power_of_2(number)
    return number.bit_length() - 1


# See https://arxiv.org/abs/2311.01512 Alg.1
def get_bit(number: int | torch.Tensor, bit_index: int) -> int | torch.Tensor:
    """Get the value of a specific bit in an integer."""
    return (number >> bit_index) & 1


def flip_bit(number: int | torch.Tensor, bit_index: int) -> int | torch.Tensor:
    """Flip the value of a specific bit in an integer."""
    return number ^ (1 << bit_index)


def flip_bits(number: int | torch.Tensor, bit_indices: list[int]) -> int | torch.Tensor:
    """Flip the values of multiple specific bits in an integer."""
    for bit_index in bit_indices:
        number = flip_bit(number, bit_index)
    return number


def insert_bit(number: int | torch.Tensor, bit_index: int, bit_value: int) -> int | torch.Tensor:
    """Insert a bit value at a specific index in an integer."""
    left = (number >> bit_index) << (bit_index + 1)
    middle = bit_value << bit_index
    right = number & ((1 << bit_index) - 1)
    return left | middle | right


def all_bits_are_one(number: int, bit_indices: list[int]) -> bool:
    """Check if all specified bits in an integer are set to 1."""
    return all(get_bit(number, i) for i in bit_indices)


def get_bit_mask(bit_indices: list[int]) -> int:
    """Get the integer representation of a bitmask with bits set at the specified indices."""
    return flip_bits(0, bit_indices)
