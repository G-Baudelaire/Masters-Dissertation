from open_factor.ansatz_handlers import AnsatzHandler
from open_factor.factorisers import HybridFactoriser
from open_factor.history_handlers import HistoryHandler
from open_factor.optimisation_handlers import OptimisationHandler


class SEFactoriser(HybridFactoriser):
    def __init__(self, ansatz_handler: AnsatzHandler, optimiser: str, history_handler: HistoryHandler, backend,
                 optimisation_shots=1024):
        super().__init__(ansatz_handler=ansatz_handler, optimisation_handler=SEOptimisationHandler(optimiser),
                         history_handler=history_handler, backend=backend, optimisation_shots=optimisation_shots)


class SEOptimisationHandler(OptimisationHandler):
    def _f_cost_function(self, n, p, q):
        return (n - p * q) ** 2
