import numpy as np
from qiskit import generate_preset_pass_manager, transpile
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


class EigensolverOptimisationHandler:
    def optimise_parameters(self, backend, ansatz, hamiltonian):
        raise NotImplementedError()


class SobhaniOptimisationHandler(EigensolverOptimisationHandler):
    def optimise_parameters(self, backend, ansatz, hamiltonian):
        pass_manager = generate_preset_pass_manager(target=backend.target, optimization_level=3)

        ansatz_isa = pass_manager.run(ansatz)
        initial_parameters = 2 * np.pi * np.random.random(ansatz.num_parameters)
        cost_history_dictionary = self._initialise_cost_history_dict()

        result = minimize(
            self._cost_function,
            initial_parameters,
            args=(backend, hamiltonian, ansatz_isa, cost_history_dictionary),
            method="cobyla",
        )

        return result, cost_history_dictionary

    def _cost_function(self, parameters, backend, hamiltonian, ansatz_isa, cost_history):
        alpha = 0.5
        shots = 1024
        bound_circuit = ansatz_isa.assign_parameters(parameters)
        bound_circuit.measure_all()
        compiled_circuit = transpile(bound_circuit, backend=backend)
        result = backend.run(compiled_circuit, shots=shots).result()
        bitstrings = list(result.get_counts().keys())
        eigenvalues = np.array([self.compute_eigenvalue(bitstring, hamiltonian).real for bitstring in bitstrings])
        energy = self.compute_cvar(eigenvalues, alpha)

        cost_history["iters"] += 1
        cost_history["prev_vector"] = parameters
        cost_history["cost_history"].append(energy)
        print(f"Iters. done: {cost_history['iters']} [Current cost: {energy}]")
        return energy

    def bit_value(self, bit, operator, term_value):
        if operator == 'Z':
            term_value *= 1 if bit == '0' else -1
        # For other Pauli operators, include appropriate mapping or basis change here.
        # e.g., for 'I' (identity), multiply by 1.
        elif operator == 'I':
            term_value *= 1
        # You can extend this if your Hamiltonian includes other operators.
        return term_value

    def compute_eigenvalue(self, bitstring, hamiltonian: SparsePauliOp):
        value = 0.0
        # Loop over each term in the Hamiltonian
        for operator, coefficient in hamiltonian.to_list():
            term_value = 1.0
            # Assume that the operator string length equals the bitstring length.
            # Map each bit: for a Pauli-Z operator, we map '0' to +1 and '1' to -1.
            for bit, bit_operator in zip(bitstring, operator):
                term_value = self.bit_value(bit, bit_operator, term_value)
            value += coefficient * term_value
        return value

    def compute_cvar(self, eigenvalues: np.ndarray, alpha):
        eigenvalues.sort()
        cutoff = max(1, int(eigenvalues.size * alpha))
        return np.mean(eigenvalues[:cutoff])

    def _initialise_cost_history_dict(self):
        return {"prev_vector": None, "iters": 0, "cost_history": []}