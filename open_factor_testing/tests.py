import csv
import os

import pandas as pd
import pytest
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider.local_service import QiskitRuntimeLocalService

from open_factor.ansatz_handlers import EfficentSU2AnsatzHandler
from open_factor.history_handlers import BlankHistoryHandler, HistoryHandler
from open_factor.squared_error_factoriser import SEFactoriser

TEST_VALUES = [
    15, 21, 57, 123, 253, 511, 1011, 2047, 4087,
    # 8189, 16379, 32743, 65509, 131069, 262099, 524281, 1048561
]
FACTORISERS = [
    # (SobhaniFactoriser, LinearCNOTAnsatzHandler(), {"optimiser": "cobyla"}, "accuracy_sobhani.csv"),
    # (InverseFactoriser, LinearCNOTAnsatzHandler(), {"optimiser": "cobyla"}, "accuracy_inverse.csv"),
    # (LogarithmicFactoriser, LinearCNOTAnsatzHandler(), {"optimiser": "cobyla"}, "accuracy_logarithmic.csv"),
    (SEFactoriser, EfficentSU2AnsatzHandler(), "cobyla", "accuracy_squared.csv"),
]
service = QiskitRuntimeLocalService
NOISE_MODELS = [
]


class Tests:
    RESULTS_PATH = "results"

    def save_to_csv(self, summary, file_name):
        out_csv = os.path.join(self.RESULTS_PATH, file_name)
        summary.to_csv(out_csv, index=False, mode='w', header=True)

    @pytest.mark.parametrize(
        "factoriser", (SEFactoriser(EfficentSU2AnsatzHandler(), "COBYLA", HistoryHandler(), AerSimulator()),)
    )
    @pytest.mark.parametrize("semiprimes", [(15, 21, 35, 77), ])
    @pytest.mark.parametrize("csv_name", ["noiseless_baseline.csv"])
    def test_noiseless_baseline(self, factoriser, semiprimes, csv_name):
        # 4. Run the baseline test
        records = []
        for n in semiprimes:
            for run_id in range(2):
                factors = factoriser.factorise(n)

                # Extract metrics
                cost_history = factoriser.get_history_handler().get_history()
                iterations = cost_history["iters"]
                success = (factors is not None)

                records.append({
                    'n': n,
                    'run': run_id,
                    'iterations': iterations,
                    "final_energy": cost_history["cost_history"][-1],
                    'success': success
                })

        # 5. Aggregate results
        df = pd.DataFrame(records)
        summary = df.groupby('n').agg(
            mean_iterations=('iterations', 'mean'),
            std_iterations=('iterations', 'std'),
            mean_final_energy=('final_energy', 'mean'),
            std_final_energy=('final_energy', 'std'),
            success_rate=('success', 'mean')
        ).reset_index()

        self.save_to_csv(summary, csv_name)
        return summary

    def optimiser_comparison(self, optimiser, n):
        pass

    def test_ansatz_circuit_depth(self, layers):
        pass

    def test_shot_dependence(self, shots):
        pass

    def test_noise_robustness(self, backend_noise_model):
        pass

    def test_scalability(self, n):
        pass

    def test_cvar_vs_standard_vqe(self, n):
        pass

    def test_real_hardware_(self, n):
        pass


class _Tests:
    @pytest.fixture(scope="module", autouse=True)
    def cleanup_results_dir(self):
        os.makedirs("results", exist_ok=True)
        yield
        # (optionally) clean up afterwards

    @pytest.mark.parametrize("factoriser_class, ansatz_handler, optimiser, csv_name", FACTORISERS)
    @pytest.mark.parametrize("n", TEST_VALUES)
    def test_factoriser_accuracy(self, factoriser_class, ansatz_handler, optimiser, csv_name, n):
        path = os.path.join("results", csv_name)
        header = ["ansatz", "optimiser", "n", "accuracy"]
        rows = self.get_result_file_rows(path, ansatz_handler, optimiser, n)
        accuracy = self.run_test(factoriser_class, ansatz_handler, optimiser, n, attempts=10)
        self.append_new_result(rows, ansatz_handler, optimiser, n, accuracy)
        self.write_to_result_file(path, header, rows)

    # @pytest.mark.parametrize("noise_model", [NoiseModel.from_backend(service.b)])
    # @pytest.mark.parametrize("n", TEST_VALUES)
    # def test_noise(self, noise_model, n):
    #     path = os.path.join("results", "noise_test.csv")
    #     header = ["ansatz", "optimiser", "n", "accuracy"]
    #     rows = self.get_result_file_rows(path, ansatz_handler, optimiser, n)
    #     accuracy = self.run_test(factoriser_class, ansatz_handler, optimiser, n, attempts=10)
    #     self.append_new_result(rows, ansatz_handler, optimiser, n, accuracy)
    #     self.write_to_result_file(path, header, rows)

    @staticmethod
    def run_test(factoriser_class, ansatz_handler, optimiser, n, attempts=1):
        fac = factoriser_class(
            ansatz_handler=ansatz_handler,
            history_handler=BlankHistoryHandler(),
            backend=AerSimulator(),
            optimiser=optimiser
        )
        successes = sum(1 for _ in range(attempts) if fac.factorise(n))
        accuracy = f"{successes / attempts:.4f}"
        return accuracy

    @staticmethod
    def get_result_file_rows(path, ansatz_handler, optimiser, n):
        rows = []
        if os.path.exists(path):
            with open(path, newline="") as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    if not (
                            row["ansatz"] == ansatz_handler.__class__.__name__ and
                            row["optimiser"] == optimiser and
                            int(row["n"]) == n
                    ):
                        rows.append(row)
        return rows

    @staticmethod
    def append_new_result(rows, ansatz_handler, optimiser, n, accuracy):
        # 3) append the fresh result
        rows.append({
            "ansatz": ansatz_handler.__class__.__name__,
            "optimiser": optimiser,
            "n": str(n),
            "accuracy": accuracy,
        })

    @staticmethod
    def write_to_result_file(path, header, rows):
        # 4) write everything back
        with open(path, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
