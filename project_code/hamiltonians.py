from itertools import product
from qiskit.quantum_info import SparsePauliOp


class HamiltonianImplementation:
    def get_hamiltonian(self, n, p_bits, q_bits, *args, **kwargs) -> SparsePauliOp:
        raise NotImplementedError()

    def _standard_projection(self, index, total_qubits):
        identity = "I" * total_qubits
        pauli = ["I"] * total_qubits
        pauli[index] = "Z"

        return 0.5 * SparsePauliOp.from_list(
            [
                (identity, 1.0),
                ("".join(pauli), -1.0)
            ]
        )


class SumOfSquaresHamiltonian(HamiltonianImplementation):
    def get_hamiltonian(self, n, p_bits, q_bits, *args, **kwargs):
        if args or kwargs:
            raise ValueError("Does not use these arguments.")

        total_qubits = p_bits + q_bits
        identity = SparsePauliOp.from_list([("I" * total_qubits, 1.0)])

        # Constant n**2
        const_term = n ** 2
        hamiltonian = const_term * identity  # Constant term
        hamiltonian += self._two_qubit_terms(n, p_bits, q_bits)  # 2-qubit term
        hamiltonian += self._four_qubit_terms(p_bits, q_bits)  # 4-qubit term
        return hamiltonian

    def _two_qubit_terms(self, n, p_bits, q_bits):
        total_qubits = p_bits + q_bits
        result = SparsePauliOp.from_list([("I" * total_qubits, 0.0)])

        for j, k in product(range(p_bits), range(q_bits)):
            coeff = -2 * n * (2 ** (j + k))
            pauli_z_pj = self._standard_projection(j, total_qubits)
            pauli_z_qk = self._standard_projection((p_bits + k), total_qubits)
            result += coeff * pauli_z_pj.compose(pauli_z_qk, front=True)
        return result

    def _four_qubit_terms(self, p_bits, q_bits):
        p_qubits, q_qubits = p_bits - 1, q_bits - 1
        total_qubits = p_qubits + q_qubits
        result = SparsePauliOp.from_list([("I" * total_qubits, 0.0)])

        for j, k, l, m in product(range(p_bits), range(q_bits), range(p_bits),
                                  range(q_bits)):
            coefficient = 2 ** (j + k + l + m)
            pauli_z_pj = self._standard_projection(j, total_qubits)
            pauli_z_qk = self._standard_projection((p_qubits + (k - 1)), total_qubits)
            pauli_z_pl = self._standard_projection(l, total_qubits)
            pauli_z_qm = self._standard_projection((p_qubits + m), total_qubits)
            result += (coefficient * pauli_z_pj.compose(pauli_z_qk, front=True)
                       .compose(pauli_z_pl, front=True)
                       .compose(pauli_z_qm, front=True))
        return result

class OptimisedSumOfSquaresHamiltonian(HamiltonianImplementation):
    def get_hamiltonian(self, n, p_bits, q_bits, *args, **kwargs):
        if args or kwargs:
            raise ValueError("Does not use these arguments.")

        p_qubits = p_bits - 1
        q_qubits = q_bits - 1
        total_qubits = p_qubits + q_qubits
        identity = SparsePauliOp.from_list([("I" * total_qubits, 1.0)])

        # Constant n**2
        const_term = n ** 2
        hamiltonian = const_term * identity  # Constant term
        hamiltonian += self._two_qubit_terms(n, p_bits, q_bits, identity)  # 2-qubit term
        hamiltonian += self._four_qubit_terms(p_bits, q_bits, identity)  # 4-qubit term
        return hamiltonian

    def _two_qubit_terms(self, n, p_bits, q_bits, identity):
        p_qubits, q_qubits = p_bits - 1, q_bits - 1
        total_qubits = p_qubits + q_qubits
        result = SparsePauliOp.from_list([("I" * total_qubits, 0.0)])

        for j, k in product(range(p_bits), range(q_bits)):
            coeff = -2 * n * (2 ** (j + k))
            pauli_z_pj = self._standard_projection((j - 1), total_qubits) if j != 0 else identity
            pauli_z_qk = self._standard_projection(((p_bits - 1) + (k - 1)), total_qubits) if k != 0 else identity
            result += coeff * pauli_z_pj.compose(pauli_z_qk, front=True)
        return result

    def _four_qubit_terms(self, p_bits, q_bits, identity):
        p_qubits, q_qubits = p_bits - 1, q_bits - 1
        total_qubits = p_qubits + q_qubits
        result = SparsePauliOp.from_list([("I" * total_qubits, 0.0)])

        for j, k, l, m in product(range(p_bits), range(q_bits), range(p_bits),
                                  range(q_bits)):
            coeff = 2 ** (j + k + l + m)
            pauli_z_pj = self._standard_projection((j - 1), total_qubits) if j != 0 else identity
            pauli_z_qk = self._standard_projection((p_qubits + (k - 1)), total_qubits) if k != 0 else identity
            pauli_z_pl = self._standard_projection((l - 1), total_qubits) if l != 0 else identity
            pauli_z_qm = self._standard_projection(p_qubits + (m - 1), total_qubits) if m != 0 else identity
            result += (coeff * pauli_z_pj.compose(pauli_z_qk, front=True)
                       .compose(pauli_z_pl, front=True)
                       .compose(pauli_z_qm, front=True))
        return result
