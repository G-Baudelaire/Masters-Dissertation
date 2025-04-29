import csv
import os
from typing import Type

import pandas as pd
import pytest
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider.local_service import QiskitRuntimeLocalService

from open_factor.ansatz_handlers import EfficentSU2AnsatzHandler, AnsatzHandler
from open_factor.history_handlers import BlankHistoryHandler, HistoryHandler
from open_factor.sobhani_factoriser import SobhaniHamiltonian, SobhaniFactoriser
from open_factor.squared_error_factoriser import SEFactoriser

TEST_VALUES = [15, 21, 57, 123, 253, 511, 1011, 2047, 4087, 8189, 16379, 32743, 65509, 131069, 262099, 524281, 1048561]
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
    SEMIPRIMES = (
        15, 21, 57, 123
    )
    NOISELESS_FACTORISERS = (
        SobhaniFactoriser(EfficentSU2AnsatzHandler(), "COBYLA", HistoryHandler(), AerSimulator()),
        SEFactoriser(EfficentSU2AnsatzHandler(), "COBYLA", HistoryHandler(), AerSimulator()),
    )

    def save_to_csv(self, summary, file_name):
        out_csv = os.path.join(self.RESULTS_PATH, file_name)
        summary.to_csv(out_csv, index=False, mode='w', header=True)

    @pytest.mark.parametrize("factoriser", NOISELESS_FACTORISERS)
    @pytest.mark.parametrize("n", SEMIPRIMES[:3])
    @pytest.mark.parametrize("csv_name", ["noiseless_baseline.csv"])
    def test_noiseless_baseline(self, factoriser, n, csv_name):
        # 4. Run the baseline test
        records = []

        for run_id in range(2):
            factors = factoriser.factorise(n)

            cost_history = factoriser.get_history_handler().get_history()
            iterations = cost_history["iters"]
            success = (factors is not None)

            records.append({
                'run': run_id,
                'iterations': iterations,
                "final_energy": cost_history["cost_history"][-1],
                'success': success
            })

        df = pd.DataFrame(records)
        summary = pd.DataFrame([{
            'factoriser': factoriser.__class__.__name__,
            'n': n,
            'mean_iterations': df['iterations'].mean(),
            'std_iterations': df['iterations'].std(),
            'mean_final_energy': df['final_energy'].mean(),
            'std_final_energy': df['final_energy'].std(),
            'success_rate': df['success'].mean()
        }])
        FileManagement.noiseless_baseline_save_to_csv(summary, csv_name)

    @pytest.mark.parametrize("optimiser", ["cobyla", "powell"])
    @pytest.mark.parametrize("n", SEMIPRIMES[:3])
    @pytest.mark.parametrize("csv_name", ["optimiser_comparison.csv"])
    def test_optimiser_comparison(self, optimiser, n, csv_name):
        records = []

        factoriser = SEFactoriser(
            EfficentSU2AnsatzHandler(),
            optimiser,
            HistoryHandler(),
            AerSimulator()
        )

        for run_id in range(1):
            factors = factoriser.factorise(n)
            # pull out history
            history_handler = factoriser.get_history_handler()
            history = history_handler.get_history()
            loop_times = factoriser.get_history_handler().get_loop_times()
            total_optimisation_time = history_handler.get_total_optimisation_time()

            # metrics
            iterations = history["iters"]
            final_energy = history["cost_history"][-1]
            mean_loop_time = total_optimisation_time / len(loop_times)
            success = (factors is not None)

            records.append({
                'n': n,
                'optimiser': optimiser,
                'run': run_id,
                'iterations': iterations,
                'final_energy': final_energy,
                'mean_loop_time': mean_loop_time,
                'success': success
            })

        df = pd.DataFrame(records)
        summary = (
            df
            .groupby(['n', 'optimiser'])
            .agg(
                mean_iterations=('iterations', 'mean'),
                std_iterations=('iterations', 'std'),
                mean_final_energy=('final_energy', 'mean'),
                std_final_energy=('final_energy', 'std'),
                mean_loop_time=('mean_loop_time', 'mean'),
                std_loop_time=('mean_loop_time', 'std'),
                success_rate=('success', 'mean')
            )
            .reset_index()
        )
        FileManagement.optimiser_save_to_csv(summary, csv_name)

    @pytest.mark.parametrize("ansatz_handler", [EfficentSU2AnsatzHandler, ])
    @pytest.mark.parametrize("n", SEMIPRIMES[:2])
    @pytest.mark.parametrize("reps", range(1, 4))
    @pytest.mark.parametrize("csv_name", ["ansatz_depth_comparison.csv"])
    def test_ansatz_circuit_depth(self, ansatz_handler: Type[AnsatzHandler], n, reps, csv_name):
        records = []

        factoriser = SEFactoriser(
            ansatz_handler(reps=reps),
            "COBYLA",
            HistoryHandler(),
            AerSimulator()
        )

        for run_id in range(5):
            factors = factoriser.factorise(n)
            # pull out history
            history_handler = factoriser.get_history_handler()
            history = history_handler.get_history()
            loop_times = factoriser.get_history_handler().get_loop_times()
            total_optimisation_time = history_handler.get_total_optimisation_time()

            # metrics
            iterations = history["iters"]
            final_energy = history["cost_history"][-1]
            mean_loop_time = total_optimisation_time / len(loop_times)
            success = (factors is not None)

            records.append({
                'ansatz':ansatz_handler.__name__,
                'reps': reps,
                'n': n,
                'run': run_id,
                'iterations': iterations,
                'final_energy': final_energy,
                'mean_loop_time': mean_loop_time,
                'success': success
            })

        df = pd.DataFrame(records)
        summary = (
            df
            .groupby(['ansatz', 'reps', 'n'])
            .agg(
                mean_iterations=('iterations', 'mean'),
                std_iterations=('iterations', 'std'),
                mean_final_energy=('final_energy', 'mean'),
                std_final_energy=('final_energy', 'std'),
                mean_loop_time=('mean_loop_time', 'mean'),
                std_loop_time=('mean_loop_time', 'std'),
                success_rate=('success', 'mean')
            )
            .reset_index()
        )
        FileManagement.ansatz_depth_save_to_csv(summary, csv_name)

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


class FileManagement:
    RESULTS_PATH = "results"

    @classmethod
    def noiseless_baseline_save_to_csv(cls, summary: pd.DataFrame, file_name: str):
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)
        out_csv = os.path.join(cls.RESULTS_PATH, file_name)

        if os.path.exists(out_csv):
            # load existing data
            existing = pd.read_csv(out_csv)

            # build a mask of rows to drop from existing:
            # any existing row whose (n, optimiser) tuple appears in summary
            to_drop = pd.concat([
                existing["factoriser"].eq(row.factoriser) & existing["n"].eq(row.n)
                for row in summary.itertuples()
            ], axis=1).any(axis=1)

            # keep only the non-matching rows
            filtered = existing.loc[~to_drop]
            # append the new summary row(s)
            merged = pd.concat([filtered, summary], ignore_index=True)
        else:
            merged = summary

        # write back, overwriting the file
        merged.to_csv(out_csv, index=False)
        print(
            f"→ Updated {out_csv} with {len(summary)} row(s) for {list(summary[['factoriser', 'n']].itertuples(index=False, name=None))}")

    @classmethod
    def optimiser_save_to_csv(cls, summary: pd.DataFrame, file_name: str):
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)
        out_csv = os.path.join(cls.RESULTS_PATH, file_name)

        if os.path.exists(out_csv):
            # load existing data
            existing = pd.read_csv(out_csv)

            # build a mask of rows to drop from existing:
            # any existing row whose (n, optimiser) tuple appears in summary
            to_drop = pd.concat([
                existing["n"].eq(row.n) & existing["optimiser"].eq(row.optimiser)
                for row in summary.itertuples()
            ], axis=1).any(axis=1)

            # keep only the non-matching rows
            filtered = existing.loc[~to_drop]
            # append the new summary row(s)
            merged = pd.concat([filtered, summary], ignore_index=True)
        else:
            merged = summary

        # write back, overwriting the file
        merged.to_csv(out_csv, index=False)
        print(
            f"→ Updated {out_csv} with {len(summary)} row(s) for {list(summary[['n', 'optimiser']].itertuples(index=False, name=None))}")

    @classmethod
    def ansatz_depth_save_to_csv(cls, summary: pd.DataFrame, file_name: str):
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)
        out_csv = os.path.join(cls.RESULTS_PATH, file_name)

        if os.path.exists(out_csv):
            # load existing data
            existing = pd.read_csv(out_csv)

            # build a mask of rows to drop from existing:
            # any existing row whose (n, optimiser) tuple appears in summary
            to_drop = pd.concat([
                existing["ansatz"].eq(row.ansatz) & existing["n"].eq(row.n)
                for row in summary.itertuples()
            ], axis=1).any(axis=1)

            # keep only the non-matching rows
            filtered = existing.loc[~to_drop]
            # append the new summary row(s)
            merged = pd.concat([filtered, summary], ignore_index=True)
        else:
            merged = summary

        # write back, overwriting the file
        merged.to_csv(out_csv, index=False)
        print(
            f"→ Updated {out_csv} with {len(summary)} row(s) for {list(summary[['n', 'reps', 'ansatz']].itertuples(index=False, name=None))}")


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
