"""Ansatze: various photonic quantum circuits"""

import copy
from itertools import combinations
from typing import Any, TYPE_CHECKING

import networkx as nx
import numpy as np
import torch
from scipy.optimize import root

from ..qmath import is_unitary
from .circuit import QumodeCircuit
from .qmath import sort_dict_fock_basis, takagi
from .state import FockState

if TYPE_CHECKING:
    from openfermion import FermionOperator


class Clements(QumodeCircuit):
    """Clements circuit."""

    def __init__(
        self,
        nmode: int,
        init_state: Any,
        cutoff: int | None = None,
        basis: bool = True,
        phi_first: bool = True,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            nmode=nmode,
            init_state=init_state,
            cutoff=cutoff,
            basis=basis,
            name='Clements',
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
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

    def dict2data(self, angle_dict: dict, dtype=torch.float) -> torch.Tensor:
        """Convert the dictionary of angles to the input data for the circuit."""
        angle_dict = angle_dict.copy()
        for key in angle_dict:
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
                        phi = angle_dict[(wire, columns[wire])]
                        theta = angle_dict[(wire, columns[wire] + 1)]
                    else:
                        theta = angle_dict[(wire, columns[wire])]
                        phi = angle_dict[(wire, columns[wire] + 1)]
                    data.append(theta)
                    data.append(phi)
                    columns[wire] += 2
            else:
                for j in range(len(wires2)):
                    wire = wires2[j] - 1
                    if self.phi_first:
                        phi = angle_dict[(wire, columns[wire])]
                        theta = angle_dict[(wire, columns[wire] + 1)]
                    else:
                        theta = angle_dict[(wire, columns[wire])]
                        phi = angle_dict[(wire, columns[wire] + 1)]
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
        cutoff: int | None = None,
        backend: str = 'gaussian',
        basis: bool = True,
        detector: str = 'pnrd',
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
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
        super().__init__(
            nmode=nmode,
            init_state='vac',
            cutoff=cutoff,
            backend=backend,
            basis=basis,
            detector=detector,
            name='GBS',
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        for i in range(self.nmode):
            self.s(i, squeezing[i])
        self.clements(unitary)


class GraphGBS(GaussianBosonSampling):
    """Simulate Gaussian Boson Sampling for graph problems."""

    def __init__(
        self,
        adj_mat: Any,
        cutoff: int | None = None,
        mean_photon_num: int | None = None,
        detector: str = 'pnrd',
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
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
        self.c = c
        lambda_c = lambd * c
        squeezing = np.arctanh(lambda_c)
        super().__init__(
            nmode=nmode,
            squeezing=squeezing,
            unitary=unitary,
            cutoff=cutoff,
            backend='gaussian',
            basis=False,
            detector=detector,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.name = 'GraphGBS'
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
    def postselect(samples: dict, nodes_list: list) -> list:
        """Postselect the results with the fixed node subgraph."""
        dic_list = [{} for _ in range(len(nodes_list))]
        for key in samples:
            temp = sum(key.state.tolist()) if isinstance(key, FockState) else sum(key)
            if temp in nodes_list:
                temp_idx = nodes_list.index(temp)
                dic_list[temp_idx][key] = samples[key]
        return dic_list

    @staticmethod
    def graph_density(graph: nx.Graph, samples: dict) -> dict:
        """Get all subgraph densities."""
        samples_ = copy.deepcopy(samples)
        for key in samples_:
            temp_prob = copy.deepcopy(samples_[key])
            if isinstance(key, FockState):
                idx = torch.nonzero(key.state).squeeze()
            else:
                idx = torch.nonzero(torch.tensor(key)).squeeze()
            density = nx.density(graph.subgraph(idx.tolist()))
            samples_[key] = [temp_prob, density]
        sort_samples = sort_dict_fock_basis(samples_, 1)
        return sort_samples


class FermionMapBoson:
    """A class to map molecular Fermionic Hamiltonians to Bosonic Qumode representations.

    This class utilizes OpenFermion and PySCF to perform electronic structure
    calculations and applies the Dhar-Mandal-Suryanarayana (DMS) mapping to
    transform the second-quantized Hamiltonian into a Bosonic Fock space.
    It supports Active Space approximations to reduce the simulation dimensionality.

    Args:
        config (dict): A configuration dictionary containing the following keys:
            'geometry' (list): Molecular structure, e.g., [('H', (0,0,0)), ('H', (0,0,0.74))].
            'basis' (str): Quantum chemistry basis set (e.g., 'sto-3g', '6-31g').
            'multiplicity' (int): Spin multiplicity (2S + 1). Usually 1 for closed-shell.
            'charge' (int): Net charge of the molecule.
            'n_ele' (int): Total number of electrons in the full system.
            'n_orbit' (int): Total number of spin-orbitals in the full space.
            'occupied_indices' (list): Indices of spatial orbitals to be frozen (occupied).
                                        Electrons in these orbitals do not participate in
                                        excitations but contribute to the energy constant.
            'active_indices' (list): Indices of spatial orbitals to be treated as active.
                                       These form the primary Hilbert space for VQE.
    """

    def __init__(self, config: dict = None) -> None:
        self.geometry = config['geometry']
        self.basis = config['basis']
        self.multiplicity = config['multiplicity']
        self.charge = config['charge']
        self.occupied_indices = config['occupied_indices']
        self.active_indices = config['active_indices']
        self.n = config['n_ele'] - 2 * len(self.occupied_indices)
        self.m = 2 * len(self.active_indices)

    def construct_h_fermion(self):
        from openfermion import get_fermion_operator
        from openfermion.chem import MolecularData
        from openfermionpyscf import run_pyscf

        molecule = MolecularData(self.geometry, self.basis, self.multiplicity, self.charge)
        molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
        hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=self.occupied_indices, active_indices=self.active_indices
        )
        fermion_op = get_fermion_operator(hamiltonian)
        h_matrix, basis_f = self.compute_hamiltonian_matrix(fermion_op, self.n, self.m)
        self.molecule = molecule
        self.hamiltonian = hamiltonian
        self.constant = hamiltonian.constant
        self.fermion_op = fermion_op
        self.basis_f = basis_f
        self.h_matrix = h_matrix
        return h_matrix

    def mapping(self):
        self.h_fock, self.map_dic = self.get_dms_mapping(self.h_matrix, self.n, self.m)
        return self.h_fock

    def fci_energy(self):
        return self.molecule.fci_energy

    @staticmethod
    def apply_annihilation(state: tuple, k: int):
        """Apply the annihilation operator :math:`f_k` to a Slater determinant.

        Args:
            state: An ordered list of occupied orbitals (p1, ..., pN) where p1 < p2 < ... < pN.
            k: The index of the orbital to be annihilated.

        """
        state = list(state)

        if k not in state:
            return None  # 轨道未被占据，结果为零

        m = state.index(k)  # k 在列表中的位置（0-indexed）
        sign = (-1) ** m  # 费米子符号
        new_state = tuple(state[:m] + state[m + 1 :])

        return (new_state, sign)

    @staticmethod
    def apply_creation(state, k):
        """Apply the creation operator :math:`f^†_k` to a Slater determinant.

        Args:
            state: An ordered list of occupied orbitals (p1, ..., pN) where p1 < p2 < ... < pN.
            k: The index of the orbital to be created.

        """
        state = list(state)

        if k in state:
            return None  # 轨道已被占据，Pauli不相容

        # 找到插入位置（保持有序）
        n = sum(1 for p in state if p < k)  # k 前面有 n 个轨道
        sign = (-1) ** n
        new_state = tuple(sorted(state + [k]))

        return (new_state, sign)

    def matrix_element_one_body(self, bra: tuple, ket: tuple, p: int, q: int):
        """Calculate the matrix element of a one-body operator: :math:`⟨bra|f^†_p f_q|ket⟩`.

        Args:
            bra: N-particle Slater determinant (bra state).
            ket: N-particle Slater determinant (ket state).
            p: Index of the creation orbital.
            q: Index of the annihilation orbital.
        """
        # Step 1: f_q 作用在 ket 上
        result = self.apply_annihilation(ket, q)
        if result is None:
            return 0.0
        int1, sign1 = result

        # Step 2: f†_p 作用在中间态上
        result = self.apply_creation(int1, p)
        if result is None:
            return 0.0
        final_state, sign2 = result

        # Step 3: 计算内积
        if final_state == bra:
            return float(sign1 * sign2)
        else:
            return 0.0

    def matrix_element_two_body(self, bra: tuple, ket: tuple, p: int, q: int, r: int, s: int):
        """Calculate the matrix element of a two-body operator: :math:`⟨bra|f†_p f†_q f_r f_s|ket⟩`.

        The operators are applied from right to left: :math:`f_s`, then :math:`f_r`,
        then :math:`f†_q`, then :math:`f†_p`.

        Args:
            bra: N-particle Slater determinant (bra state).
            ket: N-particle Slater determinant (ket state).
            p: Indices for the creation operators.
            q: Indices for the creation operators.
            r: Indices for the annihilation operators.
            s: Indices for the annihilation operators.
        """
        # Step 1: f_s 作用在 ket 上
        result = self.apply_annihilation(ket, s)
        if result is None:
            return 0.0
        int1, sign_s = result

        # Step 2: f_r 作用在 int1 上
        result = self.apply_annihilation(int1, r)
        if result is None:
            return 0.0
        int2, sign_r = result

        # Step 3: f†_q 作用在 int2 上
        result = self.apply_creation(int2, q)
        if result is None:
            return 0.0
        int3, sign_q = result

        # Step 4: f†_p 作用在 int3 上
        result = self.apply_creation(int3, p)
        if result is None:
            return 0.0
        final_state, sign_p = result

        # Step 5: 计算内积
        if final_state == bra:
            return float(sign_p * sign_q * sign_r * sign_s)
        else:
            return 0.0

    @staticmethod
    def extract_integrals(fermion_op: 'FermionOperator'):
        """Extract one-body integrals :math:`h[p,q]` and two-body integrals :math:`v[p,q,r,s]` from aFermionOperator.

        The FermionOperator format handles terms like:
            FermionOperator(0.5, '0^ 1')       -> 0.5 * f†_0 f_1
            FermionOperator(0.25, '0^ 1^ 2 3') -> 0.25 * f†_0 f†_1 f_2 f_3

        Args:
            fermion_op: The operator object from OpenFermion.
        """
        h = {}  # 单体积分
        v = {}  # 双体积分
        constant = 0.0

        for term, coeff in fermion_op.terms.items():
            if len(term) == 0:
                # 常数项
                constant += coeff.real

            elif len(term) == 2:
                # 单体项 f†_p f_q
                # term = ((p, 1), (q, 0))，其中1=产生，0=湮灭
                # 确保顺序是：产生算符在前

                # 找产生和湮灭算符
                creators = [idx for idx, dag in term if dag == 1]
                annihilators = [idx for idx, dag in term if dag == 0]

                if len(creators) == 1 and len(annihilators) == 1:
                    p, q = creators[0], annihilators[0]
                    h[(p, q)] = h.get((p, q), 0.0) + coeff.real

            elif len(term) == 4:
                # 双体项 f†_p f†_q f_r f_s
                creators = [idx for idx, dag in term if dag == 1]
                annihilators = [idx for idx, dag in term if dag == 0]

                if len(creators) == 2 and len(annihilators) == 2:
                    p, q = creators[0], creators[1]
                    # 注意：湮灭算符顺序需要与算符定义一致
                    # OpenFermion 中 f†_p f†_q f_r f_s 的 annihilators 顺序是 r, s
                    r, s = annihilators[0], annihilators[1]
                    v[(p, q, r, s)] = v.get((p, q, r, s), 0.0) + coeff.real

        return h, v, constant

    def compute_hamiltonian_matrix(self, fermion_op: 'FermionOperator', n: int, m: int):
        """Directly compute the Hamiltonian matrix in the n-particle subspace.

        Args:
            fermion_op: The input OpenFermion Hamiltonian.
            n: Number of particles (electrons).
            m: Number of spin-orbitals.
        """
        # Step 1: 枚举基矢
        basis = list(combinations(range(m), m))
        dim = len(basis)

        # Step 2: 提取积分系数
        h_integrals, v_integrals, constant = self.extract_integrals(fermion_op)

        # Step 3: 构造矩阵
        h_matrix = np.zeros((dim, dim), dtype=complex)

        # 常数项：贡献到所有对角元
        # for i in range(dim):
        #     h_matrix[i, i] += constant

        # 单体项
        for (p, q), h_pq in h_integrals.items():
            if abs(h_pq) < 1e-8:
                continue

            for j, ket in enumerate(basis):
                for i, bra in enumerate(basis):
                    me = self.matrix_element_one_body(bra, ket, p, q)
                    if me != 0.0:
                        h_matrix[i, j] += h_pq * me

        # 双体项
        for (p, q, r, s), v_pqrs in v_integrals.items():
            if abs(v_pqrs) < 1e-8:
                continue

            for j, ket in enumerate(basis):
                for i, bra in enumerate(basis):
                    me = self.matrix_element_two_body(bra, ket, p, q, r, s)
                    if me != 0.0:
                        # h_matrix[i, j] += 0.5 * v_pqrs * me
                        h_matrix[i, j] += 1 * v_pqrs * me

        return h_matrix.real, basis

    @staticmethod
    def get_dms_mapping(h_f, n, m):
        # 1. 生成所有费米子组合 (按升序)
        f_basis = list(combinations(range(m), n))

        mapping = {}
        for i, p in enumerate(f_basis):
            # 注意：p 是有序元组，如 (0, 1)
            # p[0] 是 p1, p[1] 是 p2...
            q = [0] * n
            # 应用公式 (注意索引对应关系)
            q[n - 1] = p[0]  # q_N = p1
            for j in range(1, n):  # 处理 q1 到 q_{N-1}
                # j 从 1 到 N-1 对应公式中的下标
                # p 数组索引从 0 开始，所以 p_{N-j+1} 对应 p[N-j], p_{N-j} 对应 p[N-j-1]
                q[j - 1] = p[n - j] - p[n - j - 1] - 1

            mapping[i] = ['F:', p, 'B:', tuple(q)]

        idx = h_f.nonzero()
        fock_mapping = []
        for i in range(len(idx[0])):
            k = idx[0][i]
            j = idx[1][i]
            temp = (h_f[k, j].item(), mapping[k][-1], mapping[j][-1])
            fock_mapping.append(temp)
        return fock_mapping, mapping
