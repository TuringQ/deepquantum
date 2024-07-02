"""
Ansatze: various photonic quantum circuits
"""

from typing import Any, Dict, List, Optional

import copy
import networkx as nx
import numpy as np
import torch
from scipy.optimize import root

from .circuit import QumodeCircuit
from .qmath import sort_dict_fock_basis, takagi
from ..qmath import is_unitary


class Clements(QumodeCircuit):
    """Clements circuit."""
    def __init__(
        self,
        nmode: int,
        init_state: Any,
        cutoff: Optional[int] = None,
        basis: bool = True,
        phi_first: bool = True,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(nmode=nmode, init_state=init_state, cutoff=cutoff, basis=basis, name='Clements',
                         noise=noise, mu=mu, sigma=sigma)
        self.phi_first = phi_first
        wires1 = self.wires[1::2]
        wires2 = self.wires[2::2]
        if not phi_first:
            for wire in self.wires:
                self.ps(wire, encode=True)
        for i in range(nmode):
            if i % 2 == 0:
                for j in range(len(wires1)):
                    self.mzi([wires1[j] - 1, wires1[j]], phi_first=phi_first, encode=True)
            else:
                for j in range(len(wires2)):
                    self.mzi([wires2[j] - 1, wires2[j]], phi_first=phi_first, encode=True)
        if phi_first:
            for wire in self.wires:
                self.ps(wire, encode=True)

    def dict2data(self, angle_dict: Dict, dtype = torch.float) -> torch.Tensor:
        """Convert the dictionary of angles to the input data for the circuit."""
        angle_dict = angle_dict.copy()
        for key in angle_dict.keys():
            angle = angle_dict[key]
            if not isinstance(angle, torch.Tensor):
                angle = torch.tensor(angle)
            angle_dict[key] = angle.reshape(-1)
        data = []
        columns = np.array([0] * self.nmode)
        wires1 = self.wires[1::2]
        wires2 = self.wires[2::2]
        if not self.phi_first:
            for i in range(self.nmode):
                data.append(angle_dict[(i, columns[i])])
                columns[i] += 1
        for i in range(self.nmode):
            if i % 2 == 0:
                for j in range(len(wires1)):
                    wire = wires1[j] - 1
                    if self.phi_first:
                        phi   = angle_dict[(wire, columns[wire])]
                        theta = angle_dict[(wire, columns[wire] + 1)]
                    else:
                        theta = angle_dict[(wire, columns[wire])]
                        phi   = angle_dict[(wire, columns[wire] + 1)]
                    data.append(theta)
                    data.append(phi)
                    columns[wire] += 2
            else:
                for j in range(len(wires2)):
                    wire = wires2[j] - 1
                    if self.phi_first:
                        phi   = angle_dict[(wire, columns[wire])]
                        theta = angle_dict[(wire, columns[wire] + 1)]
                    else:
                        theta = angle_dict[(wire, columns[wire])]
                        phi   = angle_dict[(wire, columns[wire] + 1)]
                    data.append(theta)
                    data.append(phi)
                    columns[wire] += 2
        if self.phi_first:
            for i in range(self.nmode):
                data.append(angle_dict[(i, columns[i])])
                columns[i] += 1
        return torch.cat(data).to(dtype)


class GaussianBosonSampling(QumodeCircuit):
    """Gaussian Boson Sampling circuit."""
    def __init__(
        self,
        nmode: int,
        squeezing: Any,
        unitary: Any,
        cutoff: Optional[int] = None,
        backend: str = 'gaussian',
        basis: bool = True,
        detector: str = 'pnrd',
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        if not isinstance(squeezing, torch.Tensor):
            squeezing = torch.tensor(squeezing).reshape(-1)
        if not isinstance(unitary, torch.Tensor):
            unitary = torch.tensor(unitary, dtype=torch.cfloat).reshape(-1, nmode)
        assert unitary.dtype in (torch.cfloat, torch.cdouble)
        assert unitary.shape[-1] == unitary.shape[-2] == nmode
        assert is_unitary(unitary)
        if cutoff is None:
            cutoff = 3
        super().__init__(nmode=nmode, init_state='vac', cutoff=cutoff, backend=backend, basis=basis,
                         detector=detector, name='GBS', noise=noise, mu=mu, sigma=sigma)
        for i in range(self.nmode):
            self.s(i, [squeezing[i], 0])
        self.clements(unitary)


class GBS_Graph(GaussianBosonSampling):
    """Simulate Gaussian Boson Sampling for graph problems."""
    def __init__(
        self,
        adj_mat: Any,
        cutoff: Optional[int] = None,
        mean_photon_num: Optional[int] = None,
        detector: str = 'pnrd',
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        if not isinstance(adj_mat, torch.Tensor):
            adj_mat = torch.tensor(adj_mat)
        assert torch.allclose(adj_mat, adj_mat.mT)
        self.adj_mat = adj_mat
        nmode = self.adj_mat.size()[-1]
        if mean_photon_num is None:
            mean_photon_num = nmode
        unitary, lambd = takagi(adj_mat)
        c = self.norm_factor_c(mean_photon_num, lambd)[0]
        lambda_c = lambd * c
        squeezing = -np.arctanh(lambda_c)
        super().__init__(nmode=nmode, squeezing=squeezing, unitary=unitary, cutoff=cutoff, backend='gaussian',
                         basis=False, detector=detector, noise=noise, mu=mu, sigma=sigma)
        self.name = 'GBS_Graph'
        self.to(adj_mat.dtype)

    @staticmethod
    def norm_factor_c(n_num, lambd, trials=20):
        """Get the normalization factor c of squeezing parameters for given total mean photon numbers."""
        lambd = np.array(lambd)

        def f(c, lambd, n_num):
            ave_n = (lambd * c) ** 2 / (1 - (lambd * c) ** 2)
            return sum(ave_n) - n_num

        sol_re = []
        for _ in range(trials):
            x_0 = np.random.uniform(0, 1 / max(lambd), 1)[0]
            re = root(f, x_0, (lambd, n_num))
            if 0 < re.x < 1 / max(lambd):
                sol_re.append(re.x[0])
        return sol_re

    @staticmethod
    def postselect(samples: Dict, nodes_list: List) -> List:
        """Postselect the results with the fixed node subgraph."""
        dic_list = [{} for _ in range(len(nodes_list))]
        for key in samples.keys():
            temp = sum(key)
            if temp in nodes_list:
                temp_idx = nodes_list.index(temp)
                dic_list[temp_idx][key] = samples[key]
        return dic_list

    @staticmethod
    def graph_density(graph: nx.Graph, samples: Dict) -> Dict:
        """Get all subgraph densities."""
        samples_ = copy.deepcopy(samples)
        for key in samples_.keys():
            temp_prob = copy.deepcopy(samples_[key])
            idx = torch.nonzero(key.state).squeeze()
            density = nx.density(graph.subgraph(idx.tolist()))
            samples_[key] = [temp_prob, density]
        sort_samples = sort_dict_fock_basis(samples_, 1)
        return sort_samples
