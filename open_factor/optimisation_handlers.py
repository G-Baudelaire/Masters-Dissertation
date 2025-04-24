from abc import abstractmethod

from qiskit.result import Result


class EigensolverOptimisationHandler:
    def __init__(self, optimiser):
        self._optimiser = optimiser

    def optimise_parameters(self, backend, ansatz, hamiltonian) -> (Result, dict):
        raise NotImplementedError()

    def _initialise_cost_history_dict(self):
        return {"prev_vector": None, "iters": 0, "cost_history": []}


class OptimisationHandler:
    def __init__(self, optimiser):
        self._optimiser = optimiser

    @abstractmethod
    def optimise_parameters(self, n, backend, ansatz) -> (Result, dict):
        raise NotImplementedError()

    def _initialise_cost_history_dict(self):
        return {"prev_vector": None, "iters": 0, "cost_history": []}

    def _update_cost_history(self, cost_history, parameters, energy):
        cost_history["iters"] += 1
        cost_history["prev_vector"] = parameters
        cost_history["cost_history"].append(energy)