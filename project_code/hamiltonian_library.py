import itertools

import numpy as np
from qiskit.quantum_info import SparsePauliOp

EXAMPLE_HAMILTONIAN = SparsePauliOp.from_list(
    [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
)


def custom_hamiltonian(number_of_qubits_p, number_of_qubits_q, number):
    total_qubits = number_of_qubits_p + number_of_qubits_q

    hamiltonian_terms = []
    coefficients = []

    # Constant term: n^2 * I
    hamiltonian_terms.append('I' * total_qubits)
    coefficients.append(number ** 2)

    # -2n * pq term

    for j, k in itertools.product(range(number_of_qubits_p), range(number_of_qubits_q)):
        coefficient = -2 * number * (2 ** (j + k))
        op_list = ['I'] * total_qubits
        op_list[j] = 'Z'
        op_list[number_of_qubits_p + k] = 'Z'
        hamiltonian_terms.append(''.join(op_list))
        coefficients.append(coefficient)

    # (pq)^2 term
    generator = itertools.product(
        range(number_of_qubits_p), range(number_of_qubits_q), range(number_of_qubits_p), range(number_of_qubits_q)
    )

    for j, k, l, m in generator:
        coefficient = 2 ** (j + k + l + m)
        op_list = ['I'] * total_qubits
        op_list[j] = 'Z'
        op_list[number_of_qubits_p + k] = 'Z'
        op_list[l] = 'Z'
        op_list[number_of_qubits_p + m] = 'Z'
        hamiltonian_terms.append(''.join(op_list))
        coefficients.append(coefficient)

    return SparsePauliOp(hamiltonian_terms, coeffs=np.array(coefficients, dtype=float))


import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def custom_hamiltonian_fixed_lsb(number_of_qubits_p, number_of_qubits_q, number):
    # LSBs are fixed, so we reduce the number of variable qubits by 1 each
    var_p = number_of_qubits_p - 1
    var_q = number_of_qubits_q - 1
    total_qubits = var_p + var_q

    hamiltonian_terms = []
    coefficients = []

    # --- CONSTANT term: n^2 ---
    hamiltonian_terms.append('I' * total_qubits)
    coefficients.append(number ** 2)

    # --- -2n * pq term ---
    # LSB contribution: 2^(0+0) = 1
    const_term = -2 * number * (2 ** 0)

    # Cross terms with variable bits
    for j, k in itertools.product(range(var_p), range(var_q)):
        coeff = -2 * number * (2 ** (j + 1 + k + 1))  # +1 to account for missing LSBs
        op_list = ['I'] * total_qubits
        op_list[j] = 'Z'
        op_list[var_p + k] = 'Z'
        hamiltonian_terms.append(''.join(op_list))
        coefficients.append(coeff)

    # Add -2n * fixed LSB (cross terms with constants)
    for j in range(var_p):
        coeff = -2 * number * (2 ** (j + 1 + 0))
        op_list = ['I'] * total_qubits
        op_list[j] = 'Z'
        hamiltonian_terms.append(''.join(op_list))
        coefficients.append(coeff)

    for k in range(var_q):
        coeff = -2 * number * (2 ** (0 + k + 1))
        op_list = ['I'] * total_qubits
        op_list[var_p + k] = 'Z'
        hamiltonian_terms.append(''.join(op_list))
        coefficients.append(coeff)

    # Add constant term from LSB * LSB
    coefficients[0] += const_term

    # --- (pq)^2 term ---

    # Constant from (LSB * LSB)^2 = 2^0 * 2^0 * 2^0 * 2^0 = 1
    coefficients[0] += 1

    # Cross terms with variable bits
    for j, k, l, m in itertools.product(range(var_p), range(var_q), range(var_p), range(var_q)):
        coeff = 2 ** (j + 1 + k + 1 + l + 1 + m + 1)
        op_list = ['I'] * total_qubits
        op_list[j] = 'Z'
        op_list[var_p + k] = 'Z'
        op_list[l] = 'Z'
        op_list[var_p + m] = 'Z'
        hamiltonian_terms.append(''.join(op_list))
        coefficients.append(coeff)

    # Mixed terms: (j,k, l/m = LSB)
    for j, k, l in itertools.product(range(var_p), range(var_q), range(var_p)):
        coeff = 2 ** (j + 1 + k + 1 + l + 1 + 0)
        op_list = ['I'] * total_qubits
        op_list[j] = 'Z'
        op_list[var_p + k] = 'Z'
        op_list[l] = 'Z'
        hamiltonian_terms.append(''.join(op_list))
        coefficients.append(coeff)

    for j, k, m in itertools.product(range(var_p), range(var_q), range(var_q)):
        coeff = 2 ** (j + 1 + k + 1 + 0 + m + 1)
        op_list = ['I'] * total_qubits
        op_list[j] = 'Z'
        op_list[var_p + k] = 'Z'
        op_list[var_p + m] = 'Z'
        hamiltonian_terms.append(''.join(op_list))
        coefficients.append(coeff)

    for j, k in itertools.product(range(var_p), range(var_q)):
        coeff = 2 ** (j + 1 + k + 1 + 0 + 0)
        op_list = ['I'] * total_qubits
        op_list[j] = 'Z'
        op_list[var_p + k] = 'Z'
        hamiltonian_terms.append(''.join(op_list))
        coefficients.append(coeff)

    # Return final SparsePauliOp
    return SparsePauliOp(hamiltonian_terms, coeffs=np.array(coefficients, dtype=float))
