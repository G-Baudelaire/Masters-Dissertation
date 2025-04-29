import time
from abc import abstractmethod

import numpy as np
from qiskit import generate_preset_pass_manager, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import Result
from scipy.optimize import minimize

from open_factor.utils import Utils


class Optimiser:
    def __init__(self, optimiser, shots=1024, max_iterations=200, tol=1e-6):
        self._optimiser = optimiser
        self._shots = shots
        self._max_iterations = max_iterations
        self._tol = tol

    def _initialise_cost_history_dict(self):
        return {"prev_vector": None, "iters": 0, "cost_history": [], "time": [time.time()]}

    def _update_cost_history(self, cost_history, parameters, energy):
        cost_history["time"].append(time.time())
        cost_history["iters"] += 1
        cost_history["prev_vector"] = parameters
        cost_history["cost_history"].append(energy)


class EigensolverOptimisationHandler(Optimiser):
    # def __init__(self, optimiser):
    #     self._optimiser = optimiser
    def optimise_parameters(self, hamiltonian, backend, ansatz):
        pass_manager = generate_preset_pass_manager(target=backend.target, optimization_level=3)

        ansatz_isa = pass_manager.run(ansatz)
        initial_parameters = 2 * np.pi * np.random.random(ansatz.num_parameters)
        cost_history_dictionary = self._initialise_cost_history_dict()
        bounds = [(0, 2 * np.pi)] * ansatz.num_parameters

        result = minimize(
            self._cost_function,
            initial_parameters,
            args=(backend, hamiltonian, ansatz_isa, cost_history_dictionary),
            method=self._optimiser,
            bounds=bounds,
            options={"maxiter": self._max_iterations},
            tol=self._tol
        )

        return result, cost_history_dictionary

    # def optimise_parameters(self, hamiltonian, backend, ansatz) -> (Result, dict):
    #     raise NotImplementedError()

    def _cost_function(self, parameters, backend, hamiltonian, ansatz_isa, cost_history):
        alpha = 1
        result = Utils.run_optimised_circuit(ansatz_isa, parameters, backend, self._shots)
        bitstrings = list(result.get_counts().keys())
        eigenvalues = np.array([self.compute_eigenvalue(bitstring, hamiltonian).real for bitstring in bitstrings])
        energy = self.compute_cvar(eigenvalues, alpha)

        self._update_cost_history(cost_history, parameters, energy)
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


class OptimisationHandler(Optimiser):
    def optimise_parameters(self, n, backend, ansatz) -> (Result, dict):
        p_bits, q_bits = Utils.bits_needed(n)
        p_qubits, q_qubits = p_bits - 1, q_bits - 1
        pass_manager = generate_preset_pass_manager(target=backend.target, optimization_level=3)

        ansatz_isa = pass_manager.run(ansatz)
        initial_parameters = 2 * np.pi * np.random.random(ansatz.num_parameters)
        cost_history_dictionary = self._initialise_cost_history_dict()
        bounds = [(0, 2 * np.pi)] * ansatz.num_parameters

        result = minimize(
            self._cost_function,
            initial_parameters,
            args=(n, p_qubits, q_qubits, backend, ansatz_isa, cost_history_dictionary),
            method=self._optimiser,
            bounds=bounds,
            options={"maxiter": self._max_iterations},
            tol=self._tol
        )

        return result, cost_history_dictionary

    def _cost_function(self, parameters, n, p_qubits, q_qubits, backend, ansatz_isa, cost_history):
        result = Utils.run_optimised_circuit(ansatz_isa, parameters, backend, self._shots)
        total = 0

        for bitstring, quantity in result.get_counts().items():
            p, q = Utils.clean_bitstring_pair(bitstring, p_qubits, q_qubits)
            score = self._f_cost_function(n, p, q)
            total += score * quantity
        energy = total / self._shots

        self._update_cost_history(cost_history, parameters, energy)
        return energy

    @abstractmethod
    def _f_cost_function(self, n, p, q):
        raise NotImplementedError()
