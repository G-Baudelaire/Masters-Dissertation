import csv
import os

import pytest
from qiskit_aer import AerSimulator

from open_factor.ansatz_handlers import LinearCNOTAnsatzHandler
from open_factor.history_handlers import BlankHistoryHandler
from open_factor.inverse_factoriser import InverseFactoriser
from open_factor.sobhani_factoriser import SobhaniFactoriser, SobhaniOptimisationHandler


class TestSobhaniFactoriser:
    ATTEMPTS = 5
    RESULTS_CSV = "sobhani_accuracy_results.csv"

    @classmethod
    def setup_class(cls):
        # if the file exists, remove it so we start fresh
        if os.path.exists(cls.RESULTS_CSV):
            os.remove(cls.RESULTS_CSV)

        # open the file and write header
        cls._csvfile = open(cls.RESULTS_CSV, "w", newline="")
        cls._writer = csv.writer(cls._csvfile)
        cls._writer.writerow(["ansatz", "optimiser", "n", "accuracy"])

    @classmethod
    def teardown_class(cls):
        cls._csvfile.close()

    @pytest.mark.parametrize(
        "ansatz_handler, optimiser, n",
        [
            (LinearCNOTAnsatzHandler(), "cobyla", 15),
            (LinearCNOTAnsatzHandler(), "powell", 15),
            # (LinearCNOTAnsatzHandler(), "cobyla", 21),
            # (LinearCNOTAnsatzHandler(), "powell", 21),
            # (LinearCNOTAnsatzHandler(), "cobyla", 77),
        ],
    )
    def test_sobhani_accuracy(self, ansatz_handler, optimiser, n):
        acc = self.run(ansatz_handler, optimiser, n)
        # append this result to our CSV
        self._writer.writerow([str(type(ansatz_handler)), optimiser, n, f"{acc:.4f}"])
        # no assert: we’re just collecting data

    def run(self, ansatz_handler, optimiser, n):
        factoriser = SobhaniFactoriser(
            ansatz_handler=ansatz_handler,
            optimiser=optimiser,
            history_handler=BlankHistoryHandler(),
            backend=AerSimulator()
        )

        successes = sum(1 for _ in range(self.ATTEMPTS) if factoriser.factorise(n))
        return successes / self.ATTEMPTS


class TestInverseFactoriser:
    ATTEMPTS = 5
    RESULTS_CSV = "inverse_accuracy_results.csv"

    @classmethod
    def setup_class(cls):
        # if the file exists, remove it so we start fresh
        if os.path.exists(cls.RESULTS_CSV):
            os.remove(cls.RESULTS_CSV)

        # open the file and write header
        cls._csvfile = open(cls.RESULTS_CSV, "w", newline="")
        cls._writer = csv.writer(cls._csvfile)
        cls._writer.writerow(["ansatz", "optimiser", "n", "accuracy"])

    @classmethod
    def teardown_class(cls):
        cls._csvfile.close()

    @pytest.mark.parametrize(
        "ansatz_handler, optimiser, n",
        [
            (LinearCNOTAnsatzHandler(), "cobyla", 15),
            (LinearCNOTAnsatzHandler(), "powell", 15),
            # (LinearCNOTAnsatzHandler(), "cobyla", 21),
            # (LinearCNOTAnsatzHandler(), "powell", 21),
            # (LinearCNOTAnsatzHandler(), "cobyla", 77),
        ],
    )
    def test_inverse_accuracy(self, ansatz_handler, optimiser, n):
        acc = self.run(ansatz_handler, optimiser, n)
        # append this result to our CSV
        self._writer.writerow([str(type(ansatz_handler)), optimiser, n, f"{acc:.4f}"])
        # no assert: we’re just collecting data

    def run(self, ansatz_handler, optimiser, n):
        factoriser = InverseFactoriser(
            ansatz_handler=ansatz_handler,
            history_handler=BlankHistoryHandler(),
            optimiser=optimiser,
            backend=AerSimulator()
        )

        successes = sum(1 for _ in range(self.ATTEMPTS) if factoriser.factorise(n))
        return successes / self.ATTEMPTS
