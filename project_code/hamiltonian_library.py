from qiskit.quantum_info import SparsePauliOp
import itertools
import numpy as np


# def custom_hamiltonian(number_of_qubits_p, number_of_qubits_q, number):
#     total_qubits = number_of_qubits_p + number_of_qubits_q
#
#     hamiltonian_terms = []
#     coefficients = []
#
#     # Constant term: n^2 * I
#     hamiltonian_terms.append('I' * total_qubits)
#     coefficients.append(number ** 2)
#
#     # -2n * pq term
#
#     for j, k in itertools.product(range(number_of_qubits_p), range(number_of_qubits_q)):
#         coefficient = -2 * number * (2 ** (j + k))
#         op_list = ['I'] * total_qubits
#         op_list[j] = 'Z'
#         op_list[number_of_qubits_p + k] = 'Z'
#         hamiltonian_terms.append(''.join(op_list))
#         coefficients.append(coefficient)
#
#     # (pq)^2 term
#     generator = itertools.product(
#         range(number_of_qubits_p), range(number_of_qubits_q), range(number_of_qubits_p), range(number_of_qubits_q)
#     )
#
#     for j, k, l, m in generator:
#         coefficient = 2 ** (j + k + l + m)
#         op_list = ['I'] * total_qubits
#         op_list[j] = 'Z'
#         op_list[number_of_qubits_p + k] = 'Z'
#         op_list[l] = 'Z'
#         op_list[number_of_qubits_p + m] = 'Z'
#         hamiltonian_terms.append(''.join(op_list))
#         coefficients.append(coefficient)
#
#     return SparsePauliOp(hamiltonian_terms, coeffs=np.array(coefficients, dtype=float))

def projector(index, total_qubits):
    identity = "I" * total_qubits
    pauli = ["I"] * total_qubits
    pauli[index] = "Z"

    return 0.5 * SparsePauliOp.from_list(
        [
            (identity, 1.0),
            ("".join(pauli), -1.0)
        ]
    )


def custom_hamiltonian(n, number_of_bits_p, number_of_bits_q):
    number_of_qubits_p = number_of_bits_p - 1
    number_of_qubits_q = number_of_bits_q - 1
    total_qubits = number_of_qubits_p + number_of_qubits_q
    identity = SparsePauliOp.from_list([("I" * total_qubits, 1.0)])

    # Constant n**2
    const_term = n ** 2
    hamiltonian = SparsePauliOp.from_list([("I" * total_qubits, const_term)])

    # 2-qubit term
    generator = itertools.product(range(number_of_qubits_p), range(number_of_qubits_q))
    for j, k in generator:
        coeff = -2 * n * (2 ** (j + k))
        pauli_z_pj = projector(j - 1, total_qubits) if j != 0 else identity
        pauli_z_qk = projector(number_of_qubits_p + k - 1, total_qubits) if k != 0 else identity
        hamiltonian += coeff * pauli_z_pj.compose(pauli_z_qk, front=True)

    # 4-qubit term
    generator = itertools.product(
        range(number_of_qubits_p), range(number_of_qubits_q), range(number_of_qubits_p), range(number_of_qubits_q)
    )

    for j, k, l, m in generator:
        coeff = 2 ** (j + k + l + m)
        pauli_z_pj = projector(j - 1, total_qubits) if j != 0 else identity
        pauli_z_qk = projector(number_of_qubits_p + k - 1, total_qubits) if k != 0 else identity
        pauli_z_pl = projector(l - 1, total_qubits) if l != 0 else identity
        pauli_z_qm = projector(number_of_qubits_p + m - 1, total_qubits) if m != 0 else identity
        hamiltonian += coeff * (pauli_z_pj.compose(pauli_z_qk, front=True)
                                .compose(pauli_z_pl, front=True)
                                .compose(pauli_z_qm, front=True))

    return hamiltonian

