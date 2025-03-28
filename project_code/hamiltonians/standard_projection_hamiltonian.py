from itertools import product
from qiskit.quantum_info import SparsePauliOp


class CustomHamiltonian:
    def get_hamiltonian(self, n, p_bits, q_bits):
        p_qubits = p_bits - 1
        q_qubits = q_bits - 1
        total_qubits = p_qubits + q_qubits
        identity = SparsePauliOp.from_list([("I" * total_qubits, 1.0)])

        # Constant n**2
        const_term = n ** 2
        hamiltonian = const_term * identity  # Constant term
        hamiltonian += self.two_qubit_terms(n, p_bits, q_bits, identity)  # 2-qubit term
        hamiltonian += self.four_qubit_terms(p_bits, q_bits, identity)  # 4-qubit term
        return hamiltonian

    def two_qubit_terms(self, n, p_bits, q_bits, identity):
        p_qubits, q_qubits = p_bits - 1, q_bits - 1
        total_qubits = p_qubits + q_qubits
        result = SparsePauliOp.from_list([("I" * total_qubits, 0.0)])

        for j, k in product(range(p_bits), range(q_bits)):
            coeff = -2 * n * (2 ** (j + k))
            pauli_z_pj = self.projector((j - 1), total_qubits) if j != 0 else identity
            pauli_z_qk = self.projector(((p_bits - 1) + (k - 1)), total_qubits) if k != 0 else identity
            result += coeff * pauli_z_pj.compose(pauli_z_qk, front=True)
        return result

    def four_qubit_terms(self, p_bits, q_bits, identity):
        p_qubits, q_qubits = p_bits - 1, q_bits - 1
        total_qubits = p_qubits + q_qubits
        result = SparsePauliOp.from_list([("I" * total_qubits, 0.0)])

        for j, k, l, m in product(range(p_bits), range(q_bits), range(p_bits),
                                  range(q_bits)):
            coeff = 2 ** (j + k + l + m)
            pauli_z_pj = self.projector((j - 1), total_qubits) if j != 0 else identity
            pauli_z_qk = self.projector((p_qubits + (k - 1)), total_qubits) if k != 0 else identity
            pauli_z_pl = self.projector((l - 1), total_qubits) if l != 0 else identity
            pauli_z_qm = self.projector(p_qubits + (m - 1), total_qubits) if m != 0 else identity
            result += (coeff * pauli_z_pj.compose(pauli_z_qk, front=True)
                       .compose(pauli_z_pl, front=True)
                       .compose(pauli_z_qm, front=True))
        return result

    def projector(self, index, total_qubits):
        identity = "I" * total_qubits
        pauli = ["I"] * total_qubits
        pauli[index] = "Z"

        return 0.5 * SparsePauliOp.from_list(
            [
                (identity, 1.0),
                ("".join(pauli), -1.0)
            ]
        )
