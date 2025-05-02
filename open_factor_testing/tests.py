import os
from typing import Type

import pandas as pd
import pytest
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from open_factor.ansatz_handlers import EfficentSU2AnsatzHandler, AnsatzHandler, NLocalAnsatzHandler, \
    RealAmplitudesAnsatzHandler
from open_factor.exponential_penalty_factoriser import ExponentialFactoriser
from open_factor.factorisers import HybridFactoriser
from open_factor.history_handlers import BlankHistoryHandler, HistoryHandler
from open_factor.inverse_factoriser import InverseFactoriser, InverseOptimisationHandler
from open_factor.logarithmic_factoriser import LogarithmicFactoriser
from open_factor.sobhani_factoriser import SobhaniHamiltonian, ExpectationSobhaniFactoriser
from open_factor.squared_error_factoriser import SEFactoriser
from qiskit_ibm_runtime import QiskitRuntimeService

from open_factor.tanh_squared_factoriser import TanhSquaredFactoriser

RESULTS_PATH = "results"
SEMIPRIMES = (
    # 15, 21, 57, 123, 253, 511, 1011,
    # 2047,
    4087, 8189,
    # 16379, 32743, 65509, 131069, 262099, 524281, 1048561
)
NOISELESS_FACTORISERS = (
    # ExpectationSobhaniFactoriser(EfficentSU2AnsatzHandler(), "COBYLA", HistoryHandler(), AerSimulator()),
    # SEFactoriser(EfficentSU2AnsatzHandler(), "COBYLA", HistoryHandler(), AerSimulator()),
    # LogarithmicFactoriser(EfficentSU2AnsatzHandler(), "COBYLA", HistoryHandler(), AerSimulator()),
    # InverseFactoriser(EfficentSU2AnsatzHandler(), "COBYLA", HistoryHandler(), AerSimulator()),
    TanhSquaredFactoriser(EfficentSU2AnsatzHandler(), "COBYLA", HistoryHandler(), AerSimulator()),
    ExponentialFactoriser(EfficentSU2AnsatzHandler(), "COBYLA", HistoryHandler(), AerSimulator()),
)
service = QiskitRuntimeService()
REAL_BACKENDS = [
    (service.backend("ibm_brisbane"), "ibm_brisbane"),
    (service.backend("ibm_sherbrooke"), "ibm_sherbrooke")
]


class Tests:
    @pytest.mark.parametrize("factoriser", NOISELESS_FACTORISERS)
    @pytest.mark.parametrize("n", SEMIPRIMES)
    @pytest.mark.parametrize("csv_name", ["noiseless_baseline.csv"])
    def test_noiseless_baseline(self, factoriser, n, csv_name):
        # 4. Run the baseline test
        records = []

        for run_id in range(50):
            factors = factoriser.factorise(n)

            history_handler = factoriser.get_history_handler()

            cost_history = history_handler.get_history()
            iterations = cost_history["iters"]
            loop_times = history_handler.get_loop_times()
            total_optimisation_time = history_handler.get_total_optimisation_time()
            mean_loop_time = total_optimisation_time / len(loop_times)

            success = (factors is not None)

            records.append({
                'run': run_id,
                'iterations': iterations,
                "final_energy": cost_history["cost_history"][-1],
                'optimisation_time': mean_loop_time,
                'success': success
            })

        df = pd.DataFrame(records)
        summary = pd.DataFrame([{
            'factoriser': factoriser.__class__.__name__,
            'n': n,
            'mean_optimisation_time':df['optimisation_time'].mean(),
            'mean_iterations': df['iterations'].mean(),
            'std_iterations': df['iterations'].std(),
            'mean_final_energy': df['final_energy'].mean(),
            'std_final_energy': df['final_energy'].std(),
            'success_rate': df['success'].mean()
        }])
        FileManagement.noiseless_baseline_save_to_csv(summary, csv_name)

    @pytest.mark.parametrize("optimiser", [
        # "COBYLA",
        "Powell"])
    @pytest.mark.parametrize("n", SEMIPRIMES)
    @pytest.mark.parametrize("csv_name", ["optimiser_comparison.csv"])
    def test_optimiser_comparison(self, optimiser, n, csv_name):
        records = []

        factoriser = SEFactoriser(
            EfficentSU2AnsatzHandler(),
            optimiser,
            HistoryHandler(),
            AerSimulator()
        )

        for run_id in range(50):
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

    @pytest.mark.parametrize("ansatz_handler", [
        EfficentSU2AnsatzHandler,
        RealAmplitudesAnsatzHandler
                             ])
    @pytest.mark.parametrize("n", SEMIPRIMES)
    @pytest.mark.parametrize("reps", range(3, 4))
    @pytest.mark.parametrize("csv_name", ["ansatz_depth_comparison.csv"])
    def test_ansatz_circuit_depth(self, ansatz_handler: Type[AnsatzHandler], n, reps, csv_name):
        records = []

        factoriser = SEFactoriser(
            ansatz_handler(reps=reps),
            "COBYLA",
            HistoryHandler(),
            AerSimulator()
        )

        for run_id in range(10):
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
                'ansatz': ansatz_handler.__name__,
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

    @pytest.mark.parametrize("optimisation_shots", [1000, 5000, 10000, 20000])
    @pytest.mark.parametrize("csv_name", ["shot_dependence.csv"])
    @pytest.mark.parametrize("n", [2047])
    def test_shot_dependence(self, optimisation_shots, csv_name, n):
        records = []

        factoriser = HybridFactoriser(
            EfficentSU2AnsatzHandler(),
            InverseOptimisationHandler("COBYLA",shots=optimisation_shots),
            HistoryHandler(),
            AerSimulator(),

        )

        for run_id in range(10):
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
                'shots': optimisation_shots,
                'n': n,
                'run': run_id,
                'iterations': iterations,
                'final_energy': final_energy,
                'mean_loop_time': mean_loop_time,
                'success': success
            })

        df = pd.DataFrame(records)
        summary = pd.DataFrame([{
            'shots': optimisation_shots,
            'n': n,
            'mean_iterations': df['iterations'].mean(),
            'std_iterations': df['iterations'].std(),
            'mean_final_energy': df['final_energy'].mean(),
            'std_final_energy': df['final_energy'].std(),
            'success_rate': df['success'].mean()
        }])

        FileManagement.shot_dependence_save_to_csv(summary, csv_name)

    @pytest.mark.parametrize("real_backend", REAL_BACKENDS[:1])
    @pytest.mark.parametrize("n", [2047])
    @pytest.mark.parametrize("csv_name", ["noise_robustness.csv"])
    def test_noise_robustness(self, real_backend, n, csv_name):
        records = []
        noise_model = NoiseModel.from_backend(real_backend[0])
        simulator = AerSimulator(noise_model=noise_model)
        # factoriser = HybridFactoriser(
        #     EfficentSU2AnsatzHandler(3), "COBYLA", HistoryHandler(), simulator, optimisation_shots=1024
        # )
        factoriser = HybridFactoriser(
            EfficentSU2AnsatzHandler(),
            InverseOptimisationHandler("COBYLA",shots=5000),
            HistoryHandler(),
            simulator,
        )

        for run_id in range(1):
            factors = factoriser.factorise(n)
            history = factoriser.get_history_handler().get_history()
            iterations = history["iters"]
            success = (factors is not None)
            records.append({
                'real_backend': real_backend[1],
                'n': n,
                'run': run_id,
                'iterations': iterations,
                'success': success
            })

        df = pd.DataFrame(records)
        summary = (
            df
            .groupby(['real_backend', 'n'])
            .agg(
                mean_iterations=('iterations', 'mean'),
                std_iterations=('iterations', 'std'),
                success_rate=('success', 'mean')
            )
            .reset_index()
        )
        FileManagement.noise_robustness_save_to_csv(summary, csv_name)



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
                existing["ansatz"].eq(row.ansatz) & existing["n"].eq(row.n) & existing["reps"].eq(row.reps)
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
            f"→ Updated {out_csv} with {len(summary)} row(s) for {list(summary[['n', 'reps', 'ansatz']].itertuples(index=False, name=None))}"
        )

    @classmethod
    def shot_dependence_save_to_csv(cls, summary, file_name):
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)
        out_csv = os.path.join(cls.RESULTS_PATH, file_name)

        if os.path.exists(out_csv):
            existing = pd.read_csv(out_csv)

            to_drop = pd.concat([
                existing["shots"].eq(row.shots) & existing["n"].eq(row.n)
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
            f"→ Updated {out_csv} with {len(summary)} row(s) for {list(summary[['shots', 'n']].itertuples(index=False, name=None))}"
        )

    @classmethod
    def noise_robustness_save_to_csv(cls, summary, file_name):
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)
        out_csv = os.path.join(cls.RESULTS_PATH, file_name)

        if os.path.exists(out_csv):
            existing = pd.read_csv(out_csv)

            to_drop = pd.concat([
                existing["real_backend"].eq(row.real_backend) & existing["n"].eq(row.n)
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
            f"→ Updated {out_csv} with {len(summary)} row(s) for {list(summary[['real_backend', 'n']].itertuples(index=False, name=None))}"
        )
