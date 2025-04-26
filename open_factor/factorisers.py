from abc import abstractmethod

from qiskit import generate_preset_pass_manager

from open_factor.ansatz_handlers import AnsatzHandler
from open_factor.history_handlers import HistoryHandler
from open_factor.optimisation_handlers import EigensolverOptimisationHandler, OptimisationHandler
from open_factor.hamiltonian_handlers import HamiltonianHandler
from open_factor.utils import Utils


class Factoriser:
    def __init__(self, ansatz_handler: AnsatzHandler, history_handler: HistoryHandler, backend):
        self._backend = backend
        self._ansatz_handler = ansatz_handler
        self._history_handler = history_handler

    @abstractmethod
    def factorise(self, n):
        raise NotImplementedError()

    def _run_circuit(self, n, ansatz, optimal_parameters):
        p_bits, q_bits = Utils.bits_needed(n)
        ansatz_isa = generate_preset_pass_manager(target=self._backend.target, optimization_level=3).run(ansatz)
        result = Utils.run_optimised_circuit(ansatz_isa, optimal_parameters, self._backend)
        factors = Utils.check_bit_strings(n, result.get_counts().keys(), p_bits - 1, q_bits - 1)
        return factors


class EigensolverFactoriser(Factoriser):
    def __init__(
            self, hamiltonian_handler: HamiltonianHandler, ansatz_handler: AnsatzHandler,
            optimisation_handler: EigensolverOptimisationHandler, history_handler: HistoryHandler, backend
    ):
        super().__init__(ansatz_handler=ansatz_handler, history_handler=history_handler, backend=backend)
        self._hamiltonian_handler = hamiltonian_handler
        self._optimisation_implementor = optimisation_handler

    def factorise(self, n):
        hamiltonian = self._hamiltonian_handler.get_hamiltonian(n)
        ansatz = self._ansatz_handler.get_ansatz(n)
        result, cost_history = self._optimisation_implementor.optimise_parameters(self._backend, ansatz, hamiltonian)
        self._history_handler.store_history(cost_history)
        return self._run_circuit(n, ansatz, result.x)


class HybridFactoriser(Factoriser):
    def __init__(self, ansatz_handler: AnsatzHandler, optimisation_handler: OptimisationHandler,
                 history_handler: HistoryHandler, backend):
        super().__init__(ansatz_handler=ansatz_handler, history_handler=history_handler, backend=backend)
        self._optimisation_implementor = optimisation_handler

    def factorise(self, n):
        ansatz = self._ansatz_handler.get_ansatz(n)
        result, cost_history = self._optimisation_implementor.optimise_parameters(n, self._backend, ansatz)
        self._history_handler.store_history(cost_history)
        return self._run_circuit(n, ansatz, result.x)
