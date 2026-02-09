"""
Base classes
"""

from torch import nn

from .state import GraphState


class Operation(nn.Module):
    """A base class for quantum operations.

    Args:
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        nodes (int, List[int] or None, optional): The indices of the nodes that the quantum operation acts on.
            Default: ``None``
    """

    def __init__(self, name: str | None = None, nodes: int | list[int] | None = None) -> None:
        super().__init__()
        self.name = name
        self.nodes = nodes
        self.npara = 0

    def _convert_indices(self, indices: int | list[int]) -> list[int]:
        """Convert and check the indices of the modes."""
        if isinstance(indices, int):
            indices = [indices]
        assert isinstance(indices, list), 'Invalid input type'
        assert all(isinstance(i, int) for i in indices), 'Invalid input type'
        assert len(set(indices)) == len(indices), 'Invalid input'
        return indices


class Command(Operation):
    """A base class for MBQC commands.

    Args:
        name (str): The name of the command.
        nodes (int or List[int]): The indices of the nodes that the command acts on.
    """

    def __init__(self, name: str, nodes: int | list[int]) -> None:
        nodes = self._convert_indices(nodes)
        super().__init__(name=name, nodes=nodes)

    def forward(self, x: GraphState) -> GraphState:
        """Perform a forward pass."""
        measure_dict = x.measure_dict
        for node in self.nodes:
            assert node not in measure_dict, f'Node {node} already measured'
        return x

    def extra_repr(self) -> str:
        return f'nodes={self.nodes}'
