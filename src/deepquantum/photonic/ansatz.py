"""
Ansatze: various photonic quantum circuits
"""

from typing import Any, Dict, List, Optional

import copy
import itertools
import networkx as nx
import numpy as np
import torch
from scipy.optimize import root
from thewalrus import hafnian, tor

from .circuit import QumodeCircuit
from .qmath import product_factorial, sub_matrix, sort_dict_fock_basis, takagi, quadrature_to_ladder, sample_sc_mcmc
from ..qmath import is_unitary
from .state import FockState, GaussianState


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
        if backend == 'gaussian':
            init_state = 'vac'
        elif backend == 'fock':
            if basis:
                init_state = [0] * nmode
            else:
                init_state = [(1, [0] * nmode)]
        if cutoff is None:
            cutoff = 3
        super().__init__(nmode=nmode, init_state=init_state, cutoff=cutoff, backend=backend, basis=basis,
                         name='GBS', noise=noise, mu=mu, sigma=sigma)
        self.detector = detector.lower()
        for i in range(self.nmode):
            self.s(i, [squeezing[i], 0])
        self.clements(unitary)

    def get_prob_gbs(self, sample: Any) -> torch.Tensor:
        """Get the probability of one sample obtained from MCMC sampling."""
        assert self.backend == 'gaussian'
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.int)
        return self._get_probs_gbs_helper(sample, cov=self.cov, detector=self.detector)[0]

    def get_probs_gbs(self, detector: Optional[str] = None) -> List:
        """Get the probabilities of the final states for GBS by different detectors.

        Args:
            detector (str, optional): Use ``'pnrd'`` for photon-number-resolving detector or
                ``'threshold'`` for threshold detector. Default: ``'pnrd'``
        """
        assert self.backend == 'gaussian'
        if self.state is None:
            return 'Run the circuit'
        cov, _ = self.state
        batch = cov.shape[0]
        if detector is None:
            detector = self.detector
        else:
            detector = detector.lower()
        if detector == 'pnrd':
            odd_basis, even_basis = self._get_odd_even_fock_basis()
            final_states = even_basis
        elif detector == 'threshold':
            final_states = torch.tensor(list(itertools.product(range(2), repeat=self.nmode)))
        probs = []
        for i in range(batch):
            probs_i = torch.tensor(self._get_probs_gbs_helper(final_states, cov[i], detector))
            keys = list(map(FockState, final_states.tolist()))
            if detector == 'pnrd':
                probs_i = torch.cat([probs_i, torch.zeros(len(odd_basis))])
                keys = list(map(FockState, torch.cat([even_basis, odd_basis]).tolist()))
            probs_dict = dict(zip(keys, probs_i))
            probs.append(probs_dict)
        return probs

    def sample(self, shots: int = 1024, detector: Optional[str] = None) -> Dict:
        """Sample the output states for GBS."""
        assert self.backend == 'gaussian'
        if self.state is None:
            return 'Run the circuit'
        if detector is None:
            detector = self.detector
        else:
            detector = detector.lower()
        cov, _ = self.state
        batch = cov.shape[0]
        sample_result = []
        key_set = set()
        for i in range(batch):
            sample_i = self._sample_mcmc(shots=shots, cov=cov[i], detector=detector, num_chain=5)
            key_set = set(sample_i.keys()) | key_set
            sample_result.append(sample_i)
        keys = list(key_set)
        all_values = []
        for i in range(batch):
            sample_i = sample_result[i]
            values_i = [sample_i[key] for key in keys]
            all_values.append(torch.tensor(values_i))
        all_values = torch.stack(all_values).mT
        keys = list(map(FockState, keys))
        sample_dict = dict(zip(keys, all_values))
        sample_dict = sort_dict_fock_basis(sample_dict)
        return sample_dict

    def _get_prob_gbs(
        self,
        final_state: torch.Tensor,
        matrix: torch.Tensor,
        det_q: torch.Tensor,
        detector: str = 'pnrd',
        purity: bool = True
    ) -> torch.Tensor:
        """Get the probability of the final state for GBS."""
        nmode = len(final_state)
        if purity and detector == 'pnrd':
            sub_mat = sub_matrix(matrix[:nmode, :nmode], final_state, final_state)
        else:
            final_state_double = torch.cat([final_state, final_state])
            sub_mat = sub_matrix(matrix, final_state_double, final_state_double)
        sub_mat = np.round(sub_mat.cpu().numpy(), 6)
        if detector == 'pnrd':
            if purity:
                haf = abs(hafnian(sub_mat)) ** 2
            else:
                haf = hafnian(sub_mat)
            return haf / (product_factorial(final_state) * torch.sqrt(det_q))
        elif detector == 'threshold':
            assert max(final_state) < 2, 'Threshold detector with maximum 1 photon'
            return tor(sub_mat) / torch.sqrt(det_q)

    def _get_probs_gbs_helper(self, final_states: torch.Tensor, cov: torch.Tensor, detector: str = 'pnrd') -> List:
        """Get the probabilities of the final states for GBS."""
        if final_states.dim() == 1:
            final_states = final_states.unsqueeze(0)
        nmode = final_states.shape[-1]
        identity = torch.eye(nmode, dtype=cov.dtype)
        cov_ladder = quadrature_to_ladder(cov)
        q = cov_ladder + torch.eye(2 * nmode) / 2
        det_q = torch.det(q)
        x_mat = torch.block_diag(identity.fliplr(), identity.fliplr()).fliplr() + 0j
        o_mat = torch.eye(2 * nmode) - torch.inverse(q)
        a_mat = x_mat @ o_mat
        if detector == 'pnrd':
            matrix = a_mat
        elif detector == 'threshold':
            matrix = o_mat
        probs = []
        stop_prob = 1 - 1e-2
        purity = GaussianState(self.state).check_purity()
        for i in range(len(final_states)):
            prob = self._get_prob_gbs(final_states[i], matrix, det_q, detector, purity)
            probs.append(abs(prob.real))
            if sum(probs) >= stop_prob:
                return probs
        return probs

    def  _get_odd_even_fock_basis(self):
        """Split the fock basis into the odd and even photon number parts."""
        max_photon = self.nmode * (self.cutoff - 1)
        odd_lst = []
        even_lst = []
        for i in range(0, max_photon + 1):
            state_tmp = torch.tensor([i] + [0] * (self.nmode - 1), dtype=torch.int)
            temp_basis = self._get_all_fock_basis(state_tmp)
            if i % 2 == 0:
                even_lst.append(temp_basis)
            else:
                odd_lst.append(temp_basis)
        return torch.cat(odd_lst), torch.cat(even_lst)

    def _generate_rand_sample(self, detector: str = 'pnrd'):
        """Generate random sample according to uniform proposal distribution."""
        if detector == 'threshold':
            sample = torch.randint(0, 2, [self.nmode])
        elif detector == 'pnrd':
            while True:
                sample = torch.randint(0, self.cutoff, [self.nmode])
                if sample.sum() % 2 == 0:
                    break
        return sample

    def _proposal_sampler(self):
        """The proposal sampler for MCMC sampling for the GBS circuit."""
        sample = self._generate_rand_sample(self.detector)
        return sample

    def _sample_mcmc(self, shots: int, cov: torch.Tensor, detector: str, num_chain: int):
        """Sample the output states for GBS via SC-MCMC method."""
        self.cov = cov
        self.detector = detector
        merged_samples = sample_sc_mcmc(prob_func=self.get_prob_gbs,
                                        proposal_sampler=self._proposal_sampler,
                                        shots=shots,
                                        num_chain=num_chain)
        return merged_samples


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
            idx = torch.nonzero(torch.tensor(key)).squeeze()
            density = nx.density(graph.subgraph(idx.tolist()))
            samples_[key] = [temp_prob, density]
        sort_samples = sort_dict_fock_basis(samples_, 1)
        return sort_samples
