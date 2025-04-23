from qiskit import transpile

from open_factor.ansatz_handlers import SobhaniAnsatzHandler, AnsatzHandler
from open_factor.history_handlers import HistoryHandler, SobhaniHistoryHandler
from open_factor.optimisation_handlers import EigensolverOptimisationHandler, SobhaniOptimisationHandler
from open_factor.hamiltonian_handlers import HamiltonianHandler, SobhaniHamiltonian
from utils import Utils


class VariationalEigensolverFactoriser:
    def __init__(
            self,
            hamiltonian_handler: HamiltonianHandler,
            ansatz_handler: AnsatzHandler,
            optimisation_handler: EigensolverOptimisationHandler,
            history_handler: HistoryHandler,
            backend
    ):
        self._hamiltonian_handler = hamiltonian_handler
        self._ansatz_handler = ansatz_handler
        self._optimisation_implementor = optimisation_handler
        self._history_handler = history_handler
        self._backend = backend

    def factorise(self, n):
        hamiltonian = self._hamiltonian_handler.get_hamiltonian(n)
        ansatz = self._ansatz_handler.get_ansatz(n)
        result, cost_history = self._optimisation_implementor.optimise_parameters(self._backend, ansatz, hamiltonian)
        self._history_handler.store_history(cost_history)
        self._run_circuit(n, ansatz, result.x)

    def _run_circuit(self, n, ansatz, optimal_parameters):
        raise NotImplementedError()


class VariationalAlgorithmFactoriser:
    def __init__(
            self,
            ansatz_handler: AnsatzHandler,
            optimisation_handler: EigensolverOptimisationHandler,
            history_handler: HistoryHandler,
            backend
    ):
        self._ansatz_handler = ansatz_handler
        self._optimisation_implementor = optimisation_handler
        self._history_handler = history_handler
        self._backend = backend

    def factorise(self, n):
        ansatz = self._ansatz_handler.get_ansatz(n)
        result, cost_history = self._optimisation_implementor.optimise_parameters(self._backend, ansatz)
        self._history_handler.store_history(cost_history)
        self._run_circuit(n, ansatz, result.x)

    def _run_circuit(self, n, ansatz, optimal_parameters):
        raise NotImplementedError()


class SobhaniFactoriser(VariationalEigensolverFactoriser):
    def __init__(self, backend):
        super().__init__(
            hamiltonian_handler=SobhaniHamiltonian(),
            ansatz_handler=SobhaniAnsatzHandler(),
            optimisation_handler=SobhaniOptimisationHandler(),
            history_handler=SobhaniHistoryHandler(),
            backend=backend
        )

    def _run_circuit(self, n, ansatz, optimal_parameters):
        ATTEMPTS = 100
        p_bits, q_bits = Utils.bits_needed(n)
        successes = 0
        solutions = set()
        for i in range(ATTEMPTS):
            result = self._run_optimised_circuit(ansatz, optimal_parameters, self._backend)
            factors = self._check_bit_strings(n, result.get_counts().keys(), p_bits - 1, q_bits - 1)

            if factors is not None:
                solutions.update(factors)
                successes += 1
        print(f"Success Rate: {(successes / ATTEMPTS) * 100} %\nSolutions: {solutions}")

    def _check_bit_strings(self, n, bit_strings, p_qubits, q_qubits):
        def clean_bitstring_pair(p_bitstring, q_bitstring):
            shift_add_one = lambda x: (int(x, 2) << 1) + 1
            return (shift_add_one(p_bitstring), shift_add_one(q_bitstring))

        pairs = (clean_bitstring_pair(bit_string[:p_qubits], bit_string[-q_qubits:]) for bit_string in bit_strings)
        for p, q in pairs:
            if p * q == n:
                return p, q
        return None

    def _run_optimised_circuit(self, ansatz, optimal_params, backend):
        bound_circuit = ansatz.assign_parameters(optimal_params)
        bound_circuit.measure_all()
        compiled_circuit = transpile(bound_circuit, backend=backend)
        return backend.run(compiled_circuit, shots=1024).result()
