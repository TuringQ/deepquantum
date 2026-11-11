"""Quantum chemistry helpers for photonic simulations.

This module builds molecular fermionic Hamiltonians with OpenFermion/PySCF and maps
fixed-particle electronic-structure problems to bosonic Fock-space representations for
photonic qumode circuits.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openfermion import FermionOperator


@dataclass(frozen=True)
class FermionToBosonConfig:
    """Structured configuration for ``FermionToBosonMapper``.

    Args:
        geometry: Molecular structure, for example
            ``[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))]``.
        basis: Quantum chemistry basis set such as ``'sto-3g'`` or ``'6-31g'``.
        multiplicity: Spin multiplicity ``2S + 1``.
        charge: Net charge of the molecule.
        n_electrons: Total number of electrons in the full system.
        occupied_indices: Indices of frozen occupied spatial orbitals.
        active_indices: Indices of active spatial orbitals.
    """

    geometry: Sequence[tuple[str, tuple[float, float, float]]]
    basis: str
    multiplicity: int
    charge: int
    n_electrons: int
    occupied_indices: Sequence[int] = ()
    active_indices: Sequence[int] = ()


class FermionToBosonMapper:
    """A class to map molecular Fermionic Hamiltonians to Bosonic Qumode representations.

    This class utilizes OpenFermion and PySCF to perform electronic structure
    calculations and applies the Dhar-Mandal-Suryanarayana (DMS) mapping to
    transform the second-quantized Hamiltonian into a Bosonic Fock space.
    It supports Active Space approximations to reduce the simulation dimensionality.

    Args:
        geometry: Molecular structure, e.g.
            ``[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))]``.
        basis: Quantum chemistry basis set (e.g. ``'sto-3g'``, ``'6-31g'``).
        multiplicity: Spin multiplicity ``2S + 1``. Usually ``1`` for closed-shell.
        charge: Net charge of the molecule.
        n_electrons: Total number of electrons in the full system.
        occupied_indices: Indices of frozen occupied spatial orbitals.
        active_indices: Indices of active spatial orbitals that define the active space.
    """

    def __init__(
        self,
        geometry: Sequence[tuple[str, tuple[float, float, float]]] | None = None,
        basis: str | None = None,
        multiplicity: int | None = None,
        charge: int | None = None,
        n_electrons: int | None = None,
        occupied_indices: Sequence[int] | None = None,
        active_indices: Sequence[int] | None = None,
    ) -> None:
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.n_electrons = n_electrons
        self.occupied_indices = occupied_indices
        self.active_indices = active_indices
        self.n = self.n_electrons - 2 * len(self.occupied_indices)
        self.m = 2 * len(self.active_indices)
        if self.n < 0:
            raise ValueError('The number of active electrons must be non-negative.')
        if self.m < self.n:
            raise ValueError('The number of active spin orbitals must be at least the number of active electrons.')

        self.config = FermionToBosonConfig(
            geometry=self.geometry,
            basis=self.basis,
            multiplicity=self.multiplicity,
            charge=self.charge,
            n_electrons=self.n_electrons,
            occupied_indices=self.occupied_indices,
            active_indices=self.active_indices,
        )

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
        self.h_fock, self.map_dic = self._get_dms_mapping(self.h_matrix, self.n, self.m)
        return self.h_fock

    def fci_energy(self):
        return self.molecule.fci_energy

    def _apply_annihilation(self, state: tuple, k: int):
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

    def _apply_creation(self, state, k):
        """Apply the creation operator :math:`f^\dagger_k` to a Slater determinant.

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
        r"""Calculate the matrix element of a one-body operator: :math:`⟨bra|f^\dagger_p f_q|ket⟩`.

        Args:
            bra: N-particle Slater determinant (bra state).
            ket: N-particle Slater determinant (ket state).
            p: Index of the creation orbital.
            q: Index of the annihilation orbital.

        Returns:
            A real-valued matrix element. In the occupation-number basis used here,
            applying :math:`f_p^\dagger f_q` to a Slater determinant can only produce
            zero or the same basis state up to a fermionic sign, so the nonzero result
            is always ``+1`` or ``-1``. This implementation therefore returns
            ``float(sign1 * sign2)`` and does not keep an imaginary component.
        """
        # Step 1: f_q acting on ket
        result = self._apply_annihilation(ket, q)
        if result is None:
            return 0.0
        int1, sign1 = result

        # Step 2: f^\dagger_p acting on int1
        result = self._apply_creation(int1, p)
        if result is None:
            return 0.0
        final_state, sign2 = result

        # Step 3: inner product
        if final_state == bra:
            return float(sign1 * sign2)
        else:
            return 0.0

    def matrix_element_two_body(self, bra: tuple, ket: tuple, p: int, q: int, r: int, s: int):
        r"""Calculate the matrix element of a two-body operator: :math:`⟨bra|f^\dagger_p f^\dagger_q f_r f_s|ket⟩`.

        The operators are applied from right to left: :math:`f_s`, then :math:`f_r`,
        then :math:`f^\dagger_q`, then :math:`f^\dagger_p`.

        Args:
            bra: N-particle Slater determinant (bra state).
            ket: N-particle Slater determinant (ket state).
            p: Indices for the creation operators.
            q: Indices for the creation operators.
            r: Indices for the annihilation operators.
            s: Indices for the annihilation operators.

        Returns:
            A real-valued matrix element. In the occupation-number basis used here,
            applying :math:`f_p^\dagger f_q^\dagger f_r f_s` to a Slater determinant
            can only produce zero or the same basis state up to a fermionic sign, so
            the nonzero result is always ``+1`` or ``-1``. This implementation
            therefore returns ``float(sign_p * sign_q * sign_r * sign_s)`` and does
            not keep an imaginary component.
        """
        # Step 1: f_s acting on ket
        result = self._apply_annihilation(ket, s)
        if result is None:
            return 0.0
        int1, sign_s = result

        # Step 2: f_r acting on int1
        result = self._apply_annihilation(int1, r)
        if result is None:
            return 0.0
        int2, sign_r = result

        # Step 3: f^\dagger_q acting on int2
        result = self._apply_creation(int2, q)
        if result is None:
            return 0.0
        int3, sign_q = result

        # Step 4: f^\dagger_p acting on int3
        result = self._apply_creation(int3, p)
        if result is None:
            return 0.0
        final_state, sign_p = result

        # Step 5: inner product
        if final_state == bra:
            return float(sign_p * sign_q * sign_r * sign_s)
        else:
            return 0.0

    def _extract_integrals(self, fermion_op: 'FermionOperator'):
        r"""Extract one-body integrals :math:`h[p,q]` and two-body integrals :math:`v[p,q,r,s]` from a FermionOperator.

        This helper assumes that ``fermion_op`` is already in normal order, with all creation operators placed before
        annihilation operators. The parsed terms therefore follow :math:`a_p^\dagger a_q` for one-body contributions
        and :math:`a_p^\dagger a_q^\dagger a_r a_s` for two-body contributions. If the input may contain unordered
        products, apply OpenFermion's ``normal_ordered`` before calling this method so that fermionic sign changes and
        constant contractions are handled correctly.

        The FermionOperator format handles terms like:
            FermionOperator(0.5, '0^ 1')       -> 0.5 * f^\dagger_0 f_1
            FermionOperator(0.25, '0^ 1^ 2 3') -> 0.25 * f^\dagger_0 f^\dagger_1 f_2 f_3

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
                # one-body term f^\dagger_p f_q
                # term = ((p, 1), (q, 0))，1=creation，0=annihilation

                creators = [idx for idx, dag in term if dag == 1]
                annihilators = [idx for idx, dag in term if dag == 0]

                if len(creators) == 1 and len(annihilators) == 1:
                    p, q = creators[0], annihilators[0]
                    h[(p, q)] = h.get((p, q), 0.0) + coeff.real

            elif len(term) == 4:
                # two-body term f^\dagger_p f^\dagger_q f_r f_s
                creators = [idx for idx, dag in term if dag == 1]
                annihilators = [idx for idx, dag in term if dag == 0]

                if len(creators) == 2 and len(annihilators) == 2:
                    p, q = creators[0], creators[1]
                    r, s = annihilators[0], annihilators[1]
                    v[(p, q, r, s)] = v.get((p, q, r, s), 0.0) + coeff.real

        return h, v, constant

    def compute_hamiltonian_matrix(self, fermion_op: 'FermionOperator', n: int, m: int):
        """Directly compute the Hamiltonian matrix in the n-particle subspace.

        Notice that 'fermion _op'is normal-ordered and the constant energy is not included
        in the calculated Hamiltonian.

        Args:
            fermion_op: The input OpenFermion Hamiltonian.
            n: Number of particles (electrons).
            m: Number of spin-orbitals.
        """
        basis = list(combinations(range(m), n))
        dim = len(basis)

        h_integrals, v_integrals, constant = self._extract_integrals(fermion_op)

        h_matrix = np.zeros((dim, dim), dtype=complex)

        # constant term is not included here
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

    def _get_dms_mapping(self, h_f, n, m):
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
