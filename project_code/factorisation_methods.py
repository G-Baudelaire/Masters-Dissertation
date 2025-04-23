from typing import Type

import numpy as np
from matplotlib import pyplot as plt
from qiskit import transpile, generate_preset_pass_manager
from scipy.optimize import minimize

from project_code import AnsatzImplementation, HamiltonianImplementation, OptimiserImplementor, LinearCNotAnsatz
from project_code.ansatzes import OptimisedLinearCNotAnsatz


class LogarithmFactorisationMethod:
    @staticmethod
    def inverse_factorise(
            n: int,
            attempts,
            backend,
            p_bits,
            q_bits):
        ansatz = OptimisedLinearCNotAnsatz().get_ansatz(p_bits, q_bits)
        result, cost_history = LogarithmFactorisationMethod.optimise_parameters(n, backend, ansatz, p_bits, q_bits)
        FactorisationMethods.display_cost_graph(cost_history)
        FactorisationMethods.run_circuit(n, attempts, result.x, p_bits, q_bits, ansatz, backend)

    @staticmethod
    def optimise_parameters(n, backend, ansatz, p_bits, q_bits):
        pass_manager = generate_preset_pass_manager(target=backend.target, optimization_level=3)
        ansatz_isa = pass_manager.run(ansatz)
        initial_parameters = 2 * np.pi * np.random.random(ansatz.num_parameters)
        cost_history_dictionary = {"prev_vector": None, "iters": 0, "cost_history": []}

        result = minimize(
            LogarithmFactorisationMethod._cost_function,
            initial_parameters,
            args=(backend, ansatz_isa, cost_history_dictionary, n, p_bits - 1, q_bits - 1),
            method="cobyla",
        )

        return result, cost_history_dictionary

    @staticmethod
    def _cost_function(parameters, backend, ansatz_isa, cost_history, n, p_qubits, q_qubits):
        shots = 1024
        bound_circuit = ansatz_isa.assign_parameters(parameters)
        bound_circuit.measure_all()
        compiled_circuit = transpile(bound_circuit, backend=backend)
        result = backend.run(compiled_circuit, shots=shots).result()
        original_bitstrings = list(result.get_counts().keys())
        cleaned_bitstrings = [LogarithmFactorisationMethod.clean_bitstring_pair(b[:p_qubits], b[-q_qubits:]) for b in
                              original_bitstrings]
        inverse_function = lambda x: np.log(np.absolute(n - (x[0] * x[1])) + 1)
        energies = list(map(inverse_function, cleaned_bitstrings))
        mean_energy = np.mean(energies)

        cost_history["iters"] += 1
        cost_history["prev_vector"] = parameters
        cost_history["cost_history"].append(mean_energy)
        print(f"Iters. done: {cost_history['iters']} [Current cost: {mean_energy}]")

        return mean_energy

    @staticmethod
    def clean_bitstring_pair(p_bitstring, q_bitstring):
        shift_add_one = lambda x: (int(x, 2) << 1) + 1
        return (shift_add_one(p_bitstring), shift_add_one(q_bitstring))


class InverseFactorisationMethod:
    @staticmethod
    def inverse_factorise(
            n: int,
            attempts,
            backend,
            p_bits,
            q_bits):
        ansatz = OptimisedLinearCNotAnsatz().get_ansatz(p_bits, q_bits)
        result, cost_history = InverseFactorisationMethod.optimise_parameters(n, backend, ansatz, p_bits, q_bits)
        FactorisationMethods.display_cost_graph(cost_history)
        FactorisationMethods.run_circuit(n, attempts, result.x, p_bits, q_bits, ansatz, backend)

    @staticmethod
    def optimise_parameters(n, backend, ansatz, p_bits, q_bits):
        pass_manager = generate_preset_pass_manager(target=backend.target, optimization_level=3)
        ansatz_isa = pass_manager.run(ansatz)
        initial_parameters = 2 * np.pi * np.random.random(ansatz.num_parameters)
        cost_history_dictionary = {"prev_vector": None, "iters": 0, "cost_history": []}

        result = minimize(
            InverseFactorisationMethod._cost_function,
            initial_parameters,
            args=(backend, ansatz_isa, cost_history_dictionary, n, p_bits - 1, q_bits - 1),
            method="cobyla",
        )

        return result, cost_history_dictionary

    @staticmethod
    def _cost_function(parameters, backend, ansatz_isa, cost_history, n, p_qubits, q_qubits):
        shots = 1024
        bound_circuit = ansatz_isa.assign_parameters(parameters)
        bound_circuit.measure_all()
        compiled_circuit = transpile(bound_circuit, backend=backend)
        result = backend.run(compiled_circuit, shots=shots).result()
        original_bitstrings = list(result.get_counts().keys())
        cleaned_bitstrings = [InverseFactorisationMethod.clean_bitstring_pair(b[:p_qubits], b[-q_qubits:]) for b in
                              original_bitstrings]
        inverse_function = lambda x: - (1 / (np.absolute(n - (x[0] * x[1])) + 0.001))
        energies = list(map(inverse_function, cleaned_bitstrings))
        mean_energy = np.mean(energies)

        cost_history["iters"] += 1
        cost_history["prev_vector"] = parameters
        cost_history["cost_history"].append(mean_energy)
        print(f"Iters. done: {cost_history['iters']} [Current cost: {mean_energy}]")

        return mean_energy

    @staticmethod
    def clean_bitstring_pair(p_bitstring, q_bitstring):
        shift_add_one = lambda x: (int(x, 2) << 1) + 1
        return (shift_add_one(p_bitstring), shift_add_one(q_bitstring))


class FactorisationMethods:
    @staticmethod
    def hamiltonian_factorise(
            n: int,
            ansatz_implementor: Type[AnsatzImplementation],
            hamiltonian_implementor: Type[HamiltonianImplementation],
            optimisation_implementor: Type[OptimiserImplementor],
            attempts,
            backend,
            p_bits,
            q_bits):
        ansatz = ansatz_implementor().get_ansatz(p_bits, q_bits)
        hamiltonian = hamiltonian_implementor().get_hamiltonian(n, p_bits, q_bits)
        result, cost_history = optimisation_implementor().optimise_parameters(backend, ansatz, hamiltonian)
        FactorisationMethods.display_cost_graph(cost_history)
        FactorisationMethods.run_circuit(n, attempts, result.x, p_bits, q_bits, ansatz, backend)

    @staticmethod
    def run_circuit(n, attempts, optimal_params, p_bits, q_bits, ansatz, backend):
        successes = 0
        solutions = set()
        for i in range(attempts):
            result = FactorisationMethods.run_optimised_circuit(ansatz, optimal_params, backend)
            factors = FactorisationMethods.check_bit_strings(n, result.get_counts().keys(), p_bits - 1, q_bits - 1)

            if factors is not None:
                solutions.update(factors)
                successes += 1
        print(f"Success Rate: {(successes / attempts) * 100} %\nSolutions: {solutions}")

    @staticmethod
    def display_cost_graph(cost_history_dict):
        fig, ax = plt.subplots()
        ax.plot(range(cost_history_dict["iters"]), cost_history_dict["cost_history"])
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        plt.draw()

    @staticmethod
    def check_bit_strings(n, bitstrings, p_qubits, q_qubits):
        def clean_bitstring_pair(p_bitstring, q_bitstring):
            shift_add_one = lambda x: (int(x, 2) << 1) + 1
            return (shift_add_one(p_bitstring), shift_add_one(q_bitstring))

        pairs = (clean_bitstring_pair(bitstring[:p_qubits], bitstring[-q_qubits:]) for bitstring in bitstrings)
        for p, q in pairs:
            if p * q == n:
                return p, q
        return None

    @staticmethod
    def run_optimised_circuit(ansatz, optimal_params, backend):
        bound_circuit = ansatz.assign_parameters(optimal_params)
        bound_circuit.measure_all()
        compiled_circuit = transpile(bound_circuit, backend=backend)
        return backend.run(compiled_circuit, shots=1024).result()
