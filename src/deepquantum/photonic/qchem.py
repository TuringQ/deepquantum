"""Quantum chemistry helpers for photonic simulations.

This module builds molecular fermionic Hamiltonians with OpenFermion/PySCF and maps
fixed-particle electronic-structure problems to bosonic Fock-space representations for
photonic qumode circuits.
"""

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openfermion import FermionOperator


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
            state: An ordered list of occupied orbitals :math:`(p1, ..., pN)` where :math:`p1 < p2 < ... < pN`.
            k: The index of the orbital to be annihilated.

        """
        state = list(state)

        if k not in state:
            return None

        m = state.index(k)
        sign = (-1) ** m
        new_state = tuple(state[:m] + state[m + 1 :])

        return (new_state, sign)

    @staticmethod
    def apply_creation(state, k):
        """Apply the creation operator :math:`f^†_k` to a Slater determinant.

        Args:
            state: An ordered list of occupied orbitals :math:`(p1, ..., pN)` where :math:`p1 < p2 < ... < pN`.
            k: The index of the orbital to be created.

        """
        state = list(state)

        if k in state:
            return None

        n = sum(1 for p in state if p < k)
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
        # Step 1: f_q acting on ket
        result = self.apply_annihilation(ket, q)
        if result is None:
            return 0.0
        int1, sign1 = result

        # Step 2: f†_p acting on int1
        result = self.apply_creation(int1, p)
        if result is None:
            return 0.0
        final_state, sign2 = result

        # Step 3: inner product
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
        # Step 1: f_s acting on ket
        result = self.apply_annihilation(ket, s)
        if result is None:
            return 0.0
        int1, sign_s = result

        # Step 2: f_r acting on int1
        result = self.apply_annihilation(int1, r)
        if result is None:
            return 0.0
        int2, sign_r = result

        # Step 3: f†_q acting on int2
        result = self.apply_creation(int2, q)
        if result is None:
            return 0.0
        int3, sign_q = result

        # Step 4: f†_p acting on int3
        result = self.apply_creation(int3, p)
        if result is None:
            return 0.0
        final_state, sign_p = result

        # Step 5: inner product
        if final_state == bra:
            return float(sign_p * sign_q * sign_r * sign_s)
        else:
            return 0.0

    @staticmethod
    def extract_integrals(fermion_op: 'FermionOperator'):
        r"""Extract one-body integrals :math:`h[p,q]` and two-body integrals :math:`v[p,q,r,s]` from a FermionOperator.

        This helper assumes that ``fermion_op`` is already in normal order, with all creation operators placed before
        annihilation operators. The parsed terms therefore follow :math:`a_p^\dagger a_q` for one-body contributions
        and :math:`a_p^\dagger a_q^\dagger a_r a_s` for two-body contributions. If the input may contain unordered
        products, apply OpenFermion's ``normal_ordered`` before calling this method so that fermionic sign changes and
        constant contractions are handled correctly.

        The FermionOperator format handles terms like:
            FermionOperator(0.5, '0^ 1')       -> 0.5 * f†_0 f_1
            FermionOperator(0.25, '0^ 1^ 2 3') -> 0.25 * f†_0 f†_1 f_2 f_3

        Args:
            fermion_op: The operator object from OpenFermion.
        """
        h = {}  # one-body
        v = {}  # two-body
        constant = 0.0

        for term, coeff in fermion_op.terms.items():
            if len(term) == 0:
                # constant
                constant += coeff.real

            elif len(term) == 2:
                # one-body term f†_p f_q
                # term = ((p, 1), (q, 0))，1=creation，0=annihilation

                creators = [idx for idx, dag in term if dag == 1]
                annihilators = [idx for idx, dag in term if dag == 0]

                if len(creators) == 1 and len(annihilators) == 1:
                    p, q = creators[0], annihilators[0]
                    h[(p, q)] = h.get((p, q), 0.0) + coeff.real

            elif len(term) == 4:
                # two-body term f†_p f†_q f_r f_s
                creators = [idx for idx, dag in term if dag == 1]
                annihilators = [idx for idx, dag in term if dag == 0]

                if len(creators) == 2 and len(annihilators) == 2:
                    p, q = creators[0], creators[1]
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
        basis = list(combinations(range(m), n))
        dim = len(basis)

        h_integrals, v_integrals, constant = self.extract_integrals(fermion_op)

        h_matrix = np.zeros((dim, dim), dtype=complex)

        # constant term is not added here
        # for i in range(dim):
        #     h_matrix[i, i] += constant

        # one-body term
        for (p, q), h_pq in h_integrals.items():
            if abs(h_pq) < 1e-8:
                continue

            for j, ket in enumerate(basis):
                for i, bra in enumerate(basis):
                    me = self.matrix_element_one_body(bra, ket, p, q)
                    if me != 0.0:
                        h_matrix[i, j] += h_pq * me

        # two-body term
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
        f_basis = list(combinations(range(m), n))
        mapping = {}
        for i, p in enumerate(f_basis):
            q = [0] * n
            q[n - 1] = p[0]  # q_N = p1
            for j in range(1, n):
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
