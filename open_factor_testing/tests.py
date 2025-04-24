# tests/test_factorisers.py
import csv
import os

import pytest
from qiskit_aer import AerSimulator

from open_factor.ansatz_handlers import LinearCNOTAnsatzHandler
from open_factor.history_handlers import BlankHistoryHandler
from open_factor.sobhani_factoriser import SobhaniFactoriser
from open_factor.inverse_factoriser import InverseFactoriser
from open_factor.logarithmic_factoriser import LogarithmicFactoriser
from open_factor.simple_squared_error_factoriser import SimpleSquaredErrorFactoriser

# all the classes you want to test,
# plus for Sobhani we have an extra optimiser param
FACTORISERS = [
    (SobhaniFactoriser, {"optimiser": "cobyla"}, "accuracy_sobhani.csv"),
    (InverseFactoriser, {}, "accuracy_inverse.csv"),
    (LogarithmicFactoriser, {}, "accuracy_logarithmic.csv"),
    (SimpleSquaredErrorFactoriser, {}, "accuracy_simple_squared.csv"),
]


@pytest.fixture(scope="module", autouse=True)
def cleanup_results_dir():
    os.makedirs("results", exist_ok=True)
    yield
    # (optionally) clean up afterwards


@pytest.mark.parametrize("FactoriserClass, extra_kwargs, csv_name", FACTORISERS)
@pytest.mark.parametrize("n", [11 * 13, 15, 21, 77])
def test_factoriser_accuracy(FactoriserClass, extra_kwargs, csv_name, n):
    """
    Run each FactoriserClass on several n values, write accuracies
    into its own CSV in ./results/.
    """
    # on first invocation create header
    path = os.path.join("results", csv_name)
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["ansatz", "optimiser", "n", "accuracy"])

        # instantiate and run
        fac = FactoriserClass(
            ansatz_handler=LinearCNOTAnsatzHandler(),
            history_handler=BlankHistoryHandler(),
            backend=AerSimulator(),
            **extra_kwargs
        )
        # say 5 attempts to estimate accuracy
        ATTEMPTS = 5
        successes = sum(1 for _ in range(ATTEMPTS) if fac.factorise(n))
        accuracy = successes / ATTEMPTS

        writer.writerow([
            FactoriserClass.__name__,
            extra_kwargs.get("optimiser", ""),
            n,
            f"{accuracy:.4f}"
        ])
    # no assert: we’re just collecting data
