from abc import abstractmethod

import numpy as np
from qiskit import generate_preset_pass_manager
from qiskit.result import Result
from scipy.optimize import minimize

from open_factor.utils import Utils


class EigensolverOptimisationHandler:
    def __init__(self, optimiser):
        self._optimiser = optimiser

    def optimise_parameters(self, hamiltonian, backend, ansatz) -> (Result, dict):
        raise NotImplementedError()

    def _initialise_cost_history_dict(self):
        return {"prev_vector": None, "iters": 0, "cost_history": []}


class OptimisationHandler:
    def __init__(self, optimiser, shots=1024, max_iterations=200, tol=1e-6):
        self._optimiser = optimiser
        self._shots = shots
        self._max_iterations = max_iterations
        self._tol = tol

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

    def _initialise_cost_history_dict(self):
        return {"prev_vector": None, "iters": 0, "cost_history": []}

    def _update_cost_history(self, cost_history, parameters, energy):
        cost_history["iters"] += 1
        cost_history["prev_vector"] = parameters
        cost_history["cost_history"].append(energy)

    @abstractmethod
    def _f_cost_function(self, n, p, q):
        raise NotImplementedError()
