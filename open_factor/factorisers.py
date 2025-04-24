from abc import abstractmethod

from open_factor.ansatz_handlers import AnsatzHandler
from open_factor.history_handlers import HistoryHandler
from open_factor.optimisation_handlers import EigensolverOptimisationHandler, OptimisationHandler
from open_factor.hamiltonian_handlers import HamiltonianHandler


class VariationalEigensolverFactoriser:
    def __init__(
            self,
            hamiltonian_handler: HamiltonianHandler,
            ansatz_handler: AnsatzHandler,
            optimisation_handler: EigensolverOptimisationHandler,
            history_handler: HistoryHandler,
            backend
    ):
        self._hamiltonian_handler = hamiltonian_handler
        self._ansatz_handler = ansatz_handler
        self._optimisation_implementor = optimisation_handler
        self._history_handler = history_handler
        self._backend = backend

    def factorise(self, n):
        hamiltonian = self._hamiltonian_handler.get_hamiltonian(n)
        ansatz = self._ansatz_handler.get_ansatz(n)
        result, cost_history = self._optimisation_implementor.optimise_parameters(self._backend, ansatz, hamiltonian)
        self._history_handler.store_history(cost_history)
        return self._run_circuit(n, ansatz, result.x)

    def _run_circuit(self, n, ansatz, optimal_parameters):
        raise NotImplementedError()


class VariationalAlgorithmFactoriser:
    def __init__(
            self,
            ansatz_handler: AnsatzHandler,
            optimisation_handler: OptimisationHandler,
            history_handler: HistoryHandler,
            backend
    ):
        self._ansatz_handler = ansatz_handler
        self._optimisation_implementor = optimisation_handler
        self._history_handler = history_handler
        self._backend = backend

    def factorise(self, n):
        ansatz = self._ansatz_handler.get_ansatz(n)
        result, cost_history = self._optimisation_implementor.optimise_parameters(n, self._backend, ansatz)
        self._history_handler.store_history(cost_history)
        return self._run_circuit(n, ansatz, result.x)

    @abstractmethod
    def _run_circuit(self, n, ansatz, optimal_parameters):
        raise NotImplementedError()
