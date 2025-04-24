import math

import numpy as np
from qiskit import generate_preset_pass_manager
from qiskit.result import Result
from scipy.optimize import minimize

from open_factor.ansatz_handlers import AnsatzHandler
from open_factor.factorisers import VariationalAlgorithmFactoriser
from open_factor.history_handlers import HistoryHandler
from open_factor.optimisation_handlers import OptimisationHandler
from open_factor.utils import Utils


class LogarithmicFactoriser(VariationalAlgorithmFactoriser):
    def __init__(self,
                 ansatz_handler: AnsatzHandler,
                 optimiser: str,
                 history_handler: HistoryHandler,
                 backend
                 ):
        super().__init__(
            ansatz_handler=ansatz_handler,
            optimisation_handler=LogarithmicOptimisationHandler(optimiser),
            history_handler=history_handler,
            backend=backend
        )

    def _run_circuit(self, n, ansatz, optimal_parameters):
        p_bits, q_bits = Utils.bits_needed(n)
        ansatz_isa = generate_preset_pass_manager(target=self._backend.target, optimization_level=3).run(ansatz)
        result = Utils.run_optimised_circuit(ansatz_isa, optimal_parameters, self._backend)
        factors = Utils.check_bit_strings(n, result.get_counts().keys(), p_bits - 1, q_bits - 1)
        return factors

class LogarithmicOptimisationHandler(OptimisationHandler):
    def optimise_parameters(self, n, backend, ansatz) -> (Result, dict):
        p_bits, q_bits = Utils.bits_needed(n)
        p_qubits, q_qubits = p_bits - 1, q_bits - 1
        pass_manager = generate_preset_pass_manager(target=backend.target, optimization_level=3)

        ansatz_isa = pass_manager.run(ansatz)
        initial_parameters = 2 * np.pi * np.random.random(ansatz.num_parameters)
        cost_history_dictionary = self._initialise_cost_history_dict()

        result = minimize(
            self._cost_function,
            initial_parameters,
            args=(n, p_qubits, q_qubits, backend, ansatz_isa, cost_history_dictionary),
            method=self._optimiser,
        )

        return result, cost_history_dictionary

    def _cost_function(self, parameters, n, p_qubits, q_qubits, backend, ansatz_isa, cost_history):
        result = Utils.run_optimised_circuit(ansatz_isa, parameters, backend)
        total = 0

        for bitstring, quantity in result.get_counts().items():
            p, q = Utils.clean_bitstring_pair(bitstring, p_qubits, q_qubits)
            score = math.log(abs(n - p * q) + 1)
            total += score * quantity
        energy = total / 1024

        self._update_cost_history(cost_history, parameters, energy)
        return energy
