import math

import numpy as np
from qiskit import generate_preset_pass_manager, transpile
from qiskit.result import Result
from scipy.optimize import minimize

from open_factor.ansatz_handlers import LinearCNOTAnsatzHandler, AnsatzHandler
from open_factor.history_handlers import EnergyGrapherHistoryHandeler, HistoryHandler
from open_factor.utils import Utils
from open_factor.factorisers import HybridFactoriser
from open_factor.optimisation_handlers import OptimisationHandler


class InverseFactoriser(HybridFactoriser):
    def __init__(self, ansatz_handler: AnsatzHandler, history_handler: HistoryHandler, optimiser: str, backend):
        super().__init__(ansatz_handler=ansatz_handler, optimisation_handler=InverseOptimisationHandler(optimiser),
                         history_handler=history_handler, backend=backend)


class InverseOptimisationHandler(OptimisationHandler):
    def __init__(self, optimiser):
        super().__init__(optimiser=optimiser)

    def _f_cost_function(self, n, p, q):
        return -1 / ((math.fabs(n - p * q)) + 0.001)
