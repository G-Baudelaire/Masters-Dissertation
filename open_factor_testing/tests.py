import csv
import os

import pytest
from qiskit_aer import AerSimulator

from open_factor.ansatz_handlers import LinearCNOTAnsatzHandler, EfficentSU2AnsatzHandler
from open_factor.history_handlers import BlankHistoryHandler
from open_factor.sobhani_factoriser import SobhaniFactoriser
from open_factor.inverse_factoriser import InverseFactoriser
from open_factor.logarithmic_factoriser import LogarithmicFactoriser
from open_factor.squared_error_factoriser import SEFactoriser

TEST_VALUES = [
    15, 21, 57,
    # 123, 253, 511, 1011, 2047, 4087, 8189, 16379, 32743, 65509, 131069, 262099, 524281, 1048561
]
FACTORISERS = [
    # (SobhaniFactoriser, LinearCNOTAnsatzHandler(), {"optimiser": "cobyla"}, "accuracy_sobhani.csv"),
    # (InverseFactoriser, LinearCNOTAnsatzHandler(), {"optimiser": "cobyla"}, "accuracy_inverse.csv"),
    # (LogarithmicFactoriser, LinearCNOTAnsatzHandler(), {"optimiser": "cobyla"}, "accuracy_logarithmic.csv"),
    (SEFactoriser, EfficentSU2AnsatzHandler(), "cobyla", "accuracy_squared.csv"),
]


@pytest.fixture(scope="module", autouse=True)
def cleanup_results_dir():
    os.makedirs("results", exist_ok=True)
    yield
    # (optionally) clean up afterwards


@pytest.mark.parametrize("FactoriserClass, ansatz_handler, optimiser, csv_name", FACTORISERS)
@pytest.mark.parametrize("n", TEST_VALUES)
def test_factoriser_accuracy(FactoriserClass, ansatz_handler, optimiser, csv_name, n):
    path = os.path.join("results", csv_name)
    header = ["ansatz", "optimiser", "n", "accuracy"]

    # 1) load existing rows, skipping any with the same key
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                if not (
                    row["ansatz"] == FactoriserClass.__name__ and
                    row["optimiser"] == optimiser and
                    int(row["n"]) == n
                ):
                    rows.append(row)

    # 2) compute your new accuracy
    ATTEMPTS = 100
    fac = FactoriserClass(
        ansatz_handler=ansatz_handler,
        history_handler=BlankHistoryHandler(),
        backend=AerSimulator(),
        optimiser=optimiser
    )
    successes = sum(1 for _ in range(ATTEMPTS) if fac.factorise(n))
    accuracy = f"{successes / ATTEMPTS:.4f}"

    # 3) append the fresh result
    rows.append({
        "ansatz":     FactoriserClass.__name__,
        "optimiser":  optimiser,
        "n":          str(n),
        "accuracy":   accuracy,
    })

    # 4) write everything back
    with open(path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)