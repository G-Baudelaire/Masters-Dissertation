from typing import Type

from matplotlib import pyplot as plt
from qiskit import transpile

from project_code.ansatzes import AnsatzImplementation, LinearCNotAnsatz
from project_code.backends_library import run_on_local_simulator
from project_code.costs import OptimiserImplementor
from project_code.hamiltonians import HamiltonianImplementation
from project_code.utils import Utils


def check_bit_strings(n, bitstrings, p_qubits, q_qubits):
    def clean_bitstring_pair(p_bitstring, q_bitstring):
        shift_add_one = lambda x: (int(x, 2) << 1) + 1
        return (shift_add_one(p_bitstring), shift_add_one(q_bitstring))

    pairs = (clean_bitstring_pair(bitstring[:p_qubits], bitstring[-q_qubits:]) for bitstring in bitstrings)
    for p, q in pairs:
        if p * q == n:
            return p, q
    return None

def run_optimised_circuit(ansatz, optimal_params, backend):
    bound_circuit = ansatz.assign_parameters(optimal_params)
    bound_circuit.measure_all()
    compiled_circuit = transpile(bound_circuit, backend=backend)
    return backend.run(compiled_circuit, shots=1024).result()


def run_circuit(n, attempts, optimal_params, p_bits, q_bits, ansatz, backend):
    successes = 0
    solutions = set()
    for i in range(attempts):
        result = run_optimised_circuit(ansatz, optimal_params, backend)
        factors = check_bit_strings(n, result.get_counts().keys(), p_bits - 1, q_bits - 1)

        if factors is not None:
            solutions.update(factors)
            successes += 1
    print(f"Success Rate: {(successes / attempts) * 100} %\nSolutions: {solutions}")


def display_cost_graph(cost_history_dict):
    fig, ax = plt.subplots()
    ax.plot(range(cost_history_dict["iters"]), cost_history_dict["cost_history"])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    plt.draw()


def factor(
        n:int,
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
    display_cost_graph(cost_history)
    run_circuit(n, attempts, result.x, p_bits, q_bits, ansatz, backend)
