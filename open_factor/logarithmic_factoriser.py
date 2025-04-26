import math

from open_factor.ansatz_handlers import AnsatzHandler
from open_factor.factorisers import HybridFactoriser
from open_factor.history_handlers import HistoryHandler
from open_factor.optimisation_handlers import OptimisationHandler


class LogarithmicFactoriser(HybridFactoriser):
    def __init__(self, ansatz_handler: AnsatzHandler, optimiser: str, history_handler: HistoryHandler, backend):
        super().__init__(ansatz_handler=ansatz_handler,optimisation_handler=LogarithmicOptimisationHandler(optimiser),history_handler=history_handler,backend=backend)


class LogarithmicOptimisationHandler(OptimisationHandler):
    def __init__(self, optimiser):
        super().__init__(optimiser=optimiser)

    def _f_cost_function(self, n, p, q):
        return math.log(abs(n - p * q) + 1)
