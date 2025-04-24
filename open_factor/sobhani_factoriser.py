from itertools import product

import numpy as np
from qiskit import transpile, generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from open_factor.ansatz_handlers import LinearCNOTAnsatzHandler, AnsatzHandler
from open_factor.factorisers import VariationalEigensolverFactoriser
from open_factor.hamiltonian_handlers import HamiltonianHandler
from open_factor.history_handlers import EnergyGrapherHistoryHandeler, HistoryHandler
from open_factor.optimisation_handlers import EigensolverOptimisationHandler
from open_factor.utils import Utils


class SobhaniFactoriser(VariationalEigensolverFactoriser):
    def __init__(
            self,
            ansatz_handler: AnsatzHandler,
            optimiser: str,
            history_handler: HistoryHandler,
            backend
    ):
        super().__init__(
            hamiltonian_handler=SobhaniHamiltonian(),
            ansatz_handler=ansatz_handler,
            optimisation_handler=SobhaniOptimisationHandler(optimiser),
            history_handler=history_handler,
            backend=backend
        )

    def _run_circuit(self, n, ansatz, optimal_parameters):
        p_bits, q_bits = Utils.bits_needed(n)
        ansatz_isa = generate_preset_pass_manager(target=self._backend.target, optimization_level=3).run(ansatz)
        result = Utils.run_optimised_circuit(ansatz_isa, optimal_parameters, self._backend)
        factors = Utils.check_bit_strings(n, result.get_counts().keys(), p_bits - 1, q_bits - 1)

        return factors

    def _run_optimised_circuit(self, ansatz, optimal_params, backend):
        bound_circuit = ansatz.assign_parameters(optimal_params)
        bound_circuit.measure_all()
        compiled_circuit = transpile(bound_circuit, backend=backend)
        return backend.run(compiled_circuit, shots=1024).result()


class SobhaniHamiltonian(HamiltonianHandler):
    def get_hamiltonian(self, n):
        p_bits, q_bits = Utils.bits_needed(n)
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


class SobhaniOptimisationHandler(EigensolverOptimisationHandler):
    def optimise_parameters(self, backend, ansatz, hamiltonian):
        pass_manager = generate_preset_pass_manager(target=backend.target, optimization_level=3)

        ansatz_isa = pass_manager.run(ansatz)
        initial_parameters = 2 * np.pi * np.random.random(ansatz.num_parameters)
        cost_history_dictionary = self._initialise_cost_history_dict()

        result = minimize(
            self._cost_function,
            initial_parameters,
            args=(backend, hamiltonian, ansatz_isa, cost_history_dictionary),
            method=self._optimiser,
        )

        return result, cost_history_dictionary

    def _cost_function(self, parameters, backend, hamiltonian, ansatz_isa, cost_history):
        alpha = 1
        shots = 1024
        bound_circuit = ansatz_isa.assign_parameters(parameters)
        bound_circuit.measure_all()
        compiled_circuit = transpile(bound_circuit, backend=backend)
        result = backend.run(compiled_circuit, shots=shots).result()
        bitstrings = list(result.get_counts().keys())
        eigenvalues = np.array([self.compute_eigenvalue(bitstring, hamiltonian).real for bitstring in bitstrings])
        energy = self.compute_cvar(eigenvalues, alpha)

        cost_history["iters"] += 1
        cost_history["prev_vector"] = parameters
        cost_history["cost_history"].append(energy)
        return energy

    def bit_value(self, bit, operator, term_value):
        if operator == 'Z':
            term_value *= 1 if bit == '0' else -1
        # For other Pauli operators, include appropriate mapping or basis change here.
        # e.g., for 'I' (identity), multiply by 1.
        elif operator == 'I':
            term_value *= 1
        # You can extend this if your Hamiltonian includes other operators.
        return term_value

    def compute_eigenvalue(self, bitstring, hamiltonian: SparsePauliOp):
        value = 0.0
        # Loop over each term in the Hamiltonian
        for operator, coefficient in hamiltonian.to_list():
            term_value = 1.0
            # Assume that the operator string length equals the bitstring length.
            # Map each bit: for a Pauli-Z operator, we map '0' to +1 and '1' to -1.
            for bit, bit_operator in zip(bitstring, operator):
                term_value = self.bit_value(bit, bit_operator, term_value)
            value += coefficient * term_value
        return value

    def compute_cvar(self, eigenvalues: np.ndarray, alpha):
        eigenvalues.sort()
        cutoff = max(1, int(eigenvalues.size * alpha))
        return np.mean(eigenvalues[:cutoff])
