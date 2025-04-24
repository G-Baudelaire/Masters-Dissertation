from qiskit.result import Result


class EigensolverOptimisationHandler:
    def optimise_parameters(self, backend, ansatz, hamiltonian) -> (Result, dict):
        raise NotImplementedError()

    def _initialise_cost_history_dict(self):
        return {"prev_vector": None, "iters": 0, "cost_history": []}


class OptimisationHandler:
    def optimise_parameters(self, n, backend, ansatz) -> (Result, dict):
        raise NotImplementedError()

    def _initialise_cost_history_dict(self):
        return {"prev_vector": None, "iters": 0, "cost_history": []}
