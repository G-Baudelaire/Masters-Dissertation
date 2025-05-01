import time
from abc import abstractmethod

import numpy as np
from qiskit import generate_preset_pass_manager, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import Result
from scipy.optimize import minimize

from open_factor.utils import Utils
from qiskit.primitives import BackendEstimatorV2

class Optimiser:
    def __init__(self, optimiser, shots=1024, max_iterations=500, tol=1):
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

class CVaREigensolverOptimisationHandler(Optimiser):
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

    def _cost_function(self, parameters, backend, hamiltonian, ansatz_isa, cost_history, alpha=0.05):
        result = Utils.run_optimised_circuit(ansatz_isa, parameters, backend, self._shots)
        bitstrings = list(result.get_counts().keys())
        eigenvalues = np.array([self.compute_eigenvalue(bitstring, hamiltonian).real for bitstring in bitstrings])
        energy = self.compute_cvar(eigenvalues, alpha)

        self._update_cost_history(cost_history, parameters, energy)
        return energy

    def bit_value(self, bit, operator):
        if operator == 'Z':
            return 1 if bit == '0' else -1
        elif operator == 'I':
            return 1
        return 0  # If operator is not handled

    def compute_eigenvalue(self, bitstring, hamiltonian: SparsePauliOp):
        value = 0.0
        for pauli_str, coeff in hamiltonian.to_list():
            term_val = np.prod([
                self.bit_value(bit, op)
                for bit, op in zip(bitstring, pauli_str)
            ])
            value += coeff.real * term_val
        return value

    def compute_cvar(self, eigenvalues: np.ndarray, alpha: float):
        sorted_vals = np.sort(eigenvalues)
        cutoff = max(1, int(len(sorted_vals) * alpha))
        return np.mean(sorted_vals[:cutoff])

class ExpectationEigensolverOptimisationHandler(Optimiser):
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
        estimator = BackendEstimatorV2(backend=backend, options={"shots": 1024})
        job = estimator.run(
            circuits=[ansatz_isa],
            observables=[hamiltonian],
            parameter_values=[parameters]
        )
        energy = job.result().values[0]


        self._update_cost_history(cost_history, parameters, energy)
        return energy

    def bit_value(self, bit, operator):
        if operator == 'Z':
            return 1 if bit == '0' else -1
        elif operator == 'I':
            return 1
        return 0  # If operator is not handled

    def compute_eigenvalue(self, bitstring, hamiltonian: SparsePauliOp):
        value = 0.0
        for pauli_str, coeff in hamiltonian.to_list():
            term_val = np.prod([
                self.bit_value(bit, op)
                for bit, op in zip(bitstring, pauli_str)
            ])
            value += coeff.real * term_val
        return value


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
